/*
 * porty.c - pyopennec._porty: an N-port admittance interface to OpenNEC,
 * written against the CPython C API (Limited API 3.12 when CMake finds the
 * SABI component; see CMakeLists.txt).
 *
 * Model.  The structure (straight wires, structural loads, ground, kernel)
 * is solved once per frequency for the port admittance matrix
 *
 *     Y[i, j] = I_i / V_j  with 1 V on port j and every other port shorted,
 *
 * so that I = Y @ V (antennaknobs' circuit layer reads it that way; NEC-2's
 * Y is reciprocal only to discretisation error, so the order matters),
 *
 * where a port is an applied-field voltage source on one segment (EX 0) and
 * I_i is the segment-centre current on port i's segment.  Everything a
 * circuit does (transmission lines, networks, phasing, feed voltages) is the
 * caller's, on Y.  For radiation and near fields the caller passes port
 * voltages v; currents superpose as sum_p v_p * x_p, where x_p is the
 * solution for 1 V on port p, so a field query never refactors.
 *
 * Engine sequence per frequency, as reporting.c's sequential processor does
 * it (execute_frequency_loop_sequential), minus EX/NT/TL/output:
 *   scale geometry to wavelengths -> ground constants (+ somnec for GN 2)
 *   -> apply_impedance_loading -> fill_interaction_matrix -> LU factor
 *   -> one multi-RHS solve for all ports -> compute_current_coefficients.
 *
 * Reentrancy.  The only process-global state OpenNEC has is in its deck
 * parser (input.c counters, strtok), the GS card's text fallback (strtok in
 * geometry.c) and a DEBUG_CONTOUR trace counter in somnec.c.  This module
 * builds GW/GE cards as numbers, never parses text and never emits GS, so it
 * touches none of them: each solve() runs on its own context_t and needs no
 * process-wide lock.  A Solution's context is mutated by far/near-field
 * queries (geometry rescaled, current coefficients rewritten), so each
 * Solution carries its own lock.
 *
 * No Python is ever called from OpenNEC: no log callback is installed
 * (OpenNEC reports from OpenMP worker threads); diagnostics are read from
 * ctx->errors after each step, with the GIL held.
 */
#include <Python.h>

#include "internals.h"

#include <stdint.h>
#include <string.h>

#if defined(HAVE_OPENBLAS) || defined(HAVE_BLAS) || defined(HAVE_MKL)
#define PORTY_LAPACK 1
/* The trailing size_t is gfortran's hidden length of the TRANS string. */
extern void zgetrf_(int *, int *, double _Complex *, int *, int *, int *);
extern void zgetrs_(char *, int *, int *, double _Complex *, int *, int *,
                    double _Complex *, int *, int *, size_t);
#else
#define PORTY_LAPACK 0
#endif

/* Build facts, set by CMakeLists.txt; the defaults keep a bare compile honest. */
#ifndef PYOPENNEC_BACKEND
#define PYOPENNEC_BACKEND "unknown"
#endif
#ifndef PYOPENNEC_OPENMP
#define PYOPENNEC_OPENMP 0
#endif

typedef double complex cplx;

static PyObject *PortyError;  /* engine failure after validation passed */
static PyObject *Cancelled;   /* cancel requested between frequencies */
static PyObject *StructureType;
static PyObject *SolutionType;

enum { GROUND_FREE = -1, GROUND_REFLECTION = 0, GROUND_PERFECT = 1, GROUND_SOMMERFELD = 2 };

typedef struct {
  int tag, nseg;
  double e1[3], e2[3], radius;
} wire_spec;

typedef struct {
  int type, tag, from, to;
  double f[3];
} load_spec;

typedef struct {
  int tag, seg;
} port_spec;

/* Everything an engine context is built from; owned by Structure and copied
 * into each Solution so the Solution outlives its Structure. */
typedef struct {
  int nw, nl, np;
  wire_spec *w;
  load_spec *l;
  port_spec *p;
  int ground;
  double epsr, sigma;
  int ek;
} spec_t;

typedef struct {
  PyObject_HEAD
  spec_t spec;
  int nseg;
  int *port_seg; /* global, 1-based */
} StructureObject;

typedef struct {
  PyObject_HEAD
  PyThread_type_lock lock;
  spec_t spec;
  context_t *ctx;
  int nf, np, nseg;
  double *freqs;
  int *port_seg;
  double *mx, *my, *mz, *mlen, *mrad; /* metres; ctx->geometry holds the scaled copy */
  cplx *raw;   /* (nf, np, nseg): solution of the matrix equation, before cabc */
  cplx *y;     /* (nf, np, np) */
  cplx *basis; /* (nf, np, nseg): segment-centre amps per volt */
  double somnec_mhz; /* frequency the Sommerfeld tables in ctx hold, 0 if none */
} SolutionObject;

/* ------------------------------------------------------------------------ */
/* spec ownership */

static void spec_free(spec_t *s)
{
  PyMem_Free(s->w);
  PyMem_Free(s->l);
  PyMem_Free(s->p);
  memset(s, 0, sizeof(*s));
}

static int spec_copy(spec_t *dst, const spec_t *src)
{
  *dst = *src;
  dst->w = NULL;
  dst->l = NULL;
  dst->p = NULL;
  dst->w = PyMem_Malloc(sizeof(wire_spec) * (size_t)(src->nw ? src->nw : 1));
  dst->l = PyMem_Malloc(sizeof(load_spec) * (size_t)(src->nl ? src->nl : 1));
  dst->p = PyMem_Malloc(sizeof(port_spec) * (size_t)(src->np ? src->np : 1));
  if (!dst->w || !dst->l || !dst->p) {
    spec_free(dst);
    return -1;
  }
  memcpy(dst->w, src->w, sizeof(wire_spec) * (size_t)src->nw);
  memcpy(dst->l, src->l, sizeof(load_spec) * (size_t)src->nl);
  memcpy(dst->p, src->p, sizeof(port_spec) * (size_t)src->np);
  return 0;
}

/* ------------------------------------------------------------------------ */
/* engine context lifecycle */

#define FREE_NULL(p) do { free((void *)(p)); (p) = NULL; } while (0)

/* destroy_context() frees the Sommerfeld tables, pattern/near-field points,
 * message lists and the NGF matrix only.  The arrays below stay allocated at
 * the pinned OpenNEC commit (reduced to what a GW-only structure without
 * sources allocates; the
 * remaining fields are freed too because a NULL free is harmless and a
 * future engine change might start allocating them). */
static void free_geometry(geometry_t *g, const geometry_t *keep)
{
#define G(field) do { if (!keep || g->field != keep->field) free((void *)g->field); g->field = NULL; } while (0)
  G(seg_end1_conn); G(seg_end2_conn); G(tag_nums); G(card_nums); G(patch_card_nums);
  G(end1_x); G(end1_y); G(end1_z); G(end2_x); G(end2_y); G(end2_z);
  G(x_center); G(y_center); G(z_center); G(half_len); G(radius);
  G(dir_cos_x); G(dir_cos_y); G(dir_cos_z);
  G(patch_x_center); G(patch_y_center); G(patch_z_center);
  G(patch_t1x); G(patch_t1y); G(patch_t1z); G(patch_t2x); G(patch_t2y); G(patch_t2z);
  G(patch_area); G(patch_normal_z);
#undef G
  for (int k = 0; k < g->errors.num_errors; k++)
    free(g->errors.errors[k].message);
  FREE_NULL(g->errors.errors);
  g->errors.num_errors = 0;
}

static void engine_destroy(context_t *ctx)
{
  if (!ctx)
    return;
  free_geometry(&ctx->ignored_geometry, &ctx->geometry);
  free_geometry(&ctx->geometry, NULL);
  current_t *c = &ctx->crnt;
  FREE_NULL(c->a_real); FREE_NULL(c->a_imag); FREE_NULL(c->b_real);
  FREE_NULL(c->b_imag); FREE_NULL(c->c_real); FREE_NULL(c->c_imag);
  FREE_NULL(c->surface_cur);
  network_context_t *n = &ctx->netcx;
  FREE_NULL(n->net_seg1); FREE_NULL(n->net_seg2); FREE_NULL(n->net_types);
  FREE_NULL(n->y11_real); FREE_NULL(n->y11_imag); FREE_NULL(n->y12_real);
  FREE_NULL(n->y12_imag); FREE_NULL(n->y22_real); FREE_NULL(n->y22_imag);
  FREE_NULL(n->exc_tag); FREE_NULL(n->exc_seg); FREE_NULL(n->exc_v); FREE_NULL(n->exc_i);
  FREE_NULL(n->exc_z); FREE_NULL(n->exc_y); FREE_NULL(n->exc_pwr);
  FREE_NULL(n->inp_tag); FREE_NULL(n->inp_seg); FREE_NULL(n->inp_v); FREE_NULL(n->inp_i);
  FREE_NULL(n->inp_z); FREE_NULL(n->inp_y); FREE_NULL(n->inp_pwr);
  FREE_NULL(ctx->save.pivot);
  segment_junction_t *j = &ctx->segj;
  FREE_NULL(j->junction_segs); FREE_NULL(j->coeff_const);
  FREE_NULL(j->coeff_sine); FREE_NULL(j->coeff_cos);
  FREE_NULL(ctx->smat.mode_matrix);
  voltage_sources_t *v = &ctx->vsorc;
  FREE_NULL(v->vsrc_segs); FREE_NULL(v->qdsrc_segs); FREE_NULL(v->qdsrc_indices);
  FREE_NULL(v->qdsrc_voltages); FREE_NULL(v->qdsrc_voltages_saved); FREE_NULL(v->vsrc_voltages);
  coupling_params_t *y = &ctx->yparm;
  FREE_NULL(y->pair_tags); FREE_NULL(y->pair_segs); FREE_NULL(y->y11); FREE_NULL(y->y12);
  impedance_loading_t *z = &ctx->zload;
  FREE_NULL(z->load_types); FREE_NULL(z->load_tags); FREE_NULL(z->load_tag_from);
  FREE_NULL(z->load_tag_to); FREE_NULL(z->ldcard_num); FREE_NULL(z->load_r);
  FREE_NULL(z->load_l); FREE_NULL(z->load_c); FREE_NULL(z->load_freq);
  FREE_NULL(z->seg_impedance);
  FREE_NULL(ctx->loading_outputs.entries);
  FREE_NULL(ctx->source_filename);
  destroy_context(ctx);
}

/* First diagnostic at or above PROBLEM recorded since index `from`. */
static const char *engine_problem(const context_t *ctx, int from)
{
  for (int k = from; k < ctx->errors.num_errors; k++)
    if (ctx->errors.errors[k].severity >= PROBLEM)
      return ctx->errors.errors[k].message;
  return NULL;
}

static int append_card(deck_t *deck, const char *code, const int *iv, int ni,
                       const double *fv, int nf)
{
  card_t c;
  memset(&c, 0, sizeof(c));
  c.card_code[0] = code[0];
  c.card_code[1] = code[1];
  for (int k = 0; k < ni; k++)
    c.i[k + 1] = iv[k];
  for (int k = 0; k < nf; k++)
    c.f[k + 1] = fv[k];
  c.ints_used = MAX_INT_FIELDS;
  c.flts_used = MAX_FLT_FIELDS;
  c.orig_str = strdup(code);
  c.card_str = strdup(code);
  if (!c.orig_str || !c.card_str) {
    free(c.orig_str);
    free(c.card_str);
    return -1;
  }
  return insert_card(deck, &c, deck->num_cards) == 0 ? 0 : -1;
}

/* Build a context holding the metre geometry (GW cards, then GE with the
 * ground-plane flag set when there is a ground, so wire ends on z = 0 are
 * joined to their image), the ground selection and the load table.  On
 * failure returns NULL with `err` filled.  No Python calls: runs with or
 * without the GIL. */
static context_t *engine_build(const spec_t *s, char *err, size_t errlen)
{
  context_t *ctx = create_context();
  if (!ctx) {
    snprintf(err, errlen, "out of memory creating an OpenNEC context");
    return NULL;
  }
  deck_t deck;
  init_deck(&deck);
  errors_list_t gerr = {0};
  int ok = 0;

  for (int k = 0; k < s->nw; k++) {
    const wire_spec *w = &s->w[k];
    int iv[2] = {w->tag, w->nseg};
    double fv[7] = {w->e1[0], w->e1[1], w->e1[2], w->e2[0], w->e2[1], w->e2[2], w->radius};
    if (append_card(&deck, "GW", iv, 2, fv, 7) != 0) {
      snprintf(err, errlen, "out of memory building the geometry");
      goto done;
    }
  }
  int ge = (s->ground == GROUND_FREE) ? 0 : 1;
  if (append_card(&deck, "GE", &ge, 1, NULL, 0) != 0) {
    snprintf(err, errlen, "out of memory building the geometry");
    goto done;
  }

  calculate_geometry(ctx, &deck, &gerr, &ctx->outputs);
  {
    const char *m = gerr.num_errors > 0 ? gerr.errors[0].message
                    : ctx->geometry.errors.num_errors > 0 ? ctx->geometry.errors.errors[0].message
                    : engine_problem(ctx, 0);
    if (m) {
      snprintf(err, errlen, "geometry: %s", m);
      goto done;
    }
  }
  if (ctx->geometry.num_segs <= 0 || ctx->geometry.num_patches != 0 ||
      ctx->geometry.num_segs_sym != ctx->geometry.num_segs) {
    snprintf(err, errlen, "geometry: expected plain wire segments without symmetry, the "
             "engine built %d segments, %d patches", ctx->geometry.num_segs,
             ctx->geometry.num_patches);
    goto done;
  }

  if (s->ground == GROUND_FREE) {
    ctx->gnd.has_ground = 1;
    ctx->gnd.is_perfect = 1; /* as GN -1 sets it */
  } else {
    ctx->gnd.has_ground = 2;
    ctx->gnd.is_perfect = s->ground;
    ctx->save.ground_epsr = s->epsr;
    ctx->save.ground_sigma = s->sigma;
  }
  ctx->gnd.num_radials = 0;
  ctx->gnd.far_field_type = 0;

  int n = ctx->geometry.num_segs;
  ctx->netcx.num_eq = n;
  ctx->netcx.num_eq_sym = n;

  if (s->nl > 0) {
    impedance_loading_t *z = &ctx->zload;
    size_t ni = (size_t)s->nl * sizeof(int), nd = (size_t)s->nl * sizeof(double);
    z->load_types = malloc(ni); z->load_tags = malloc(ni);
    z->load_tag_from = malloc(ni); z->load_tag_to = malloc(ni); z->ldcard_num = malloc(ni);
    z->load_r = malloc(nd); z->load_l = malloc(nd); z->load_c = malloc(nd); z->load_freq = malloc(nd);
    if (!z->load_types || !z->load_tags || !z->load_tag_from || !z->load_tag_to ||
        !z->ldcard_num || !z->load_r || !z->load_l || !z->load_c || !z->load_freq) {
      snprintf(err, errlen, "out of memory building the load table");
      goto done;
    }
    for (int k = 0; k < s->nl; k++) {
      const load_spec *l = &s->l[k];
      z->load_types[k] = l->type;
      z->load_tags[k] = l->tag;
      z->load_tag_from[k] = l->from;
      z->load_tag_to[k] = (l->to == 0) ? l->from : l->to; /* process_ld_card's rule */
      z->ldcard_num[k] = k + 1;
      z->load_r[k] = l->f[0];
      z->load_l[k] = l->f[1];
      z->load_c[k] = l->f[2];
      z->load_freq[k] = 0.0;
    }
    z->num_loads = s->nl;
  }

  size_t nb = (size_t)n * sizeof(double);
  current_t *c = &ctx->crnt;
  c->a_real = malloc(nb); c->a_imag = malloc(nb); c->b_real = malloc(nb);
  c->b_imag = malloc(nb); c->c_real = malloc(nb); c->c_imag = malloc(nb);
  c->surface_cur = malloc((size_t)n * sizeof(cplx));
  if (!c->a_real || !c->a_imag || !c->b_real || !c->b_imag || !c->c_real ||
      !c->c_imag || !c->surface_cur) {
    snprintf(err, errlen, "out of memory allocating current arrays");
    goto done;
  }
  ok = 1;

done:
  for (int k = 0; k < gerr.num_errors; k++)
    free(gerr.errors[k].message);
  free(gerr.errors);
  destroy_deck(&deck);
  if (!ok) {
    engine_destroy(ctx);
    return NULL;
  }
  return ctx;
}

/* ------------------------------------------------------------------------ */
/* per-frequency engine steps (no Python; called with the GIL released) */

/* Geometry to wavelengths with the same products as reporting.c's
 * scale_geometry_for_frequency(), then the ground constants as its
 * setup_ground_for_frequency() derives them.  The Sommerfeld tables are
 * refilled only when `want_somnec` (matrix fill, near fields); the far field
 * uses the reflection coefficients alone. */
static void engine_set_frequency(SolutionObject *s, double fmhz, int want_somnec)
{
  context_t *ctx = s->ctx;
  geometry_t *g = &ctx->geometry;
  double fr = fmhz / CVEL;
  for (int i = 0; i < s->nseg; i++) {
    g->x_center[i] = s->mx[i] * fr;
    g->y_center[i] = s->my[i] * fr;
    g->z_center[i] = s->mz[i] * fr;
    g->half_len[i] = s->mlen[i] * fr;
    g->radius[i] = s->mrad[i] * fr;
  }
  g->wavelength = CVEL / fmhz;
  ctx->save.freq_mhz = fmhz;
  ctx->dataj.k_half_len = 1.0; /* KH default */
  ctx->dataj.use_extended_kernel = s->spec.ek;

  if (ctx->gnd.has_ground == 1)
    return;
  ctx->gnd.fresnel_ratio = CPLX_10;
  if (ctx->gnd.is_perfect == 1)
    return;
  /* sigma >= 0 is enforced at validation, so the engine's negative-sigma
   * rewrite (which overwrites save.ground_sigma) never applies */
  cplx epsc = s->spec.epsr - I * s->spec.sigma * g->wavelength * 59.96;
  ctx->gnd.impedance_ratio = 1.0 / csqrt(epsc);
  ctx->gwav.impedance_ratio = ctx->gnd.impedance_ratio;
  ctx->gwav.impedance_ratio_sq = ctx->gwav.impedance_ratio * ctx->gwav.impedance_ratio;
  if (ctx->gnd.is_perfect == 2) {
    if (want_somnec && s->somnec_mhz != fmhz) {
      somnec(ctx, s->spec.epsr, s->spec.sigma, fmhz);
      /* interpolate_sommerfeld_grid() keeps the last 4x4 cell's polynomial
       * coefficients and reuses them when the next point falls in the same
       * cell; nothing invalidates them when somnec() refills the tables, so
       * a context that changes frequency would interpolate the first points
       * from the previous frequency's grid.  Reset to context_init()'s
       * state, which is what a freshly initialised context starts
       * from. */
      ctx->intrp.ixs = -10;
      ctx->intrp.iys = -10;
      ctx->intrp.igrs = -10;
      ctx->intrp.dx = 1.0;
      ctx->intrp.dy = 1.0;
      s->somnec_mhz = fmhz;
    }
    ctx->gnd.fresnel_ratio = (epsc - 1.0) / (epsc + 1.0);
  }
}

/* LU-factor the filled matrix in place.  fill_interaction_matrix() leaves
 * the transpose of the system matrix in `cm` (column-major, leading
 * dimension n).  OpenNEC's LAPACK factor_matrix() copies it into an n x n
 * scratch buffer, transposes the copy and runs zgetrf on it; for an
 * unsymmetrised structure (one section, leading dimension n) doing the
 * transpose and zgetrf in place is the same arithmetic on the same layout
 * without the second n^2 buffer.  `ip` gets LAPACK's 1-based pivots, which
 * is also what OpenNEC's solve() hands to zgetrs. */
static int engine_factor(context_t *ctx, int n, cplx *cm, int *ip)
{
#if PORTY_LAPACK
  (void)ctx;
  for (int i = 1; i < n; i++)
    for (int j = 0; j < i; j++) {
      cplx t = cm[i + (size_t)j * n];
      cm[i + (size_t)j * n] = cm[j + (size_t)i * n];
      cm[j + (size_t)i * n] = t;
    }
  int info = 0;
  zgetrf_(&n, &n, cm, &n, ip, &info);
  return info; /* > 0: exactly singular U; < 0: bad argument */
#else
  factor_matrix_symmetric(ctx, n, n, cm, ip);
  return 0;
#endif
}

/* Solve for all port excitations at once.  OpenNEC's solve() takes one
 * right-hand side and, on the LAPACK path, copies the whole n x n factored
 * matrix into a scratch buffer per call, so P ports through it would cost P
 * n^2 copies and allocations; one zgetrs with nrhs = P costs none. */
static int engine_solve(context_t *ctx, int n, int nrhs, cplx *cm, int *ip, cplx *b)
{
#if PORTY_LAPACK
  (void)ctx;
  char trans = 'N';
  int info = 0;
  zgetrs_(&trans, &n, &nrhs, cm, &n, ip, b, &n, &info, 1);
  return info;
#else
  for (int p = 0; p < nrhs; p++)
    solve(ctx, n, cm, ip, b + (size_t)p * n, n);
  return 0;
#endif
}

static int solve_frequency(SolutionObject *s, int k, cplx *cm, int *ip, cplx *rhs,
                           char *err, size_t errlen)
{
  context_t *ctx = s->ctx;
  int n = s->nseg, np = s->np;
  int e0 = ctx->errors.num_errors;
  const char *m;

  engine_set_frequency(s, s->freqs[k], 1);
  double wl = ctx->geometry.wavelength;

  if (ctx->zload.num_loads > 0 &&
      apply_impedance_loading(ctx, ctx->zload.load_types, ctx->zload.load_tags,
                              ctx->zload.load_tag_from, ctx->zload.load_tag_to,
                              ctx->zload.load_r, ctx->zload.load_l, ctx->zload.load_c) != 0) {
    m = engine_problem(ctx, e0);
    snprintf(err, errlen, "loading at %g MHz: %s", s->freqs[k], m ? m : "failed");
    return -1;
  }
  if (fill_interaction_matrix(ctx, n, cm, 1.0, s->spec.ek) != 0) {
    m = engine_problem(ctx, e0);
    snprintf(err, errlen, "matrix fill at %g MHz: %s", s->freqs[k], m ? m : "failed");
    return -1;
  }
  int info = engine_factor(ctx, n, cm, ip);
  if (info != 0) {
    snprintf(err, errlen, "LU factorisation at %g MHz: zgetrf info=%d", s->freqs[k], info);
    return -1;
  }

  /* EX 0 applied field: E = -V / (segment length), the length in metres
   * being half_len (NEC-2's SI, a FULL length in wavelengths) times lambda;
   * fill_excitation_vector() writes the same expression. */
  memset(rhs, 0, sizeof(cplx) * (size_t)n * np);
  for (int p = 0; p < np; p++) {
    int is = s->port_seg[p] - 1;
    rhs[(size_t)p * n + is] = -1.0 / (ctx->geometry.half_len[is] * wl);
  }
  info = engine_solve(ctx, n, np, cm, ip, rhs);
  if (info != 0) {
    snprintf(err, errlen, "LU solve at %g MHz: zgetrs info=%d", s->freqs[k], info);
    return -1;
  }
  memcpy(s->raw + (size_t)k * np * n, rhs, sizeof(cplx) * (size_t)n * np);

  /* compute_current_coefficients() turns basis amplitudes into segment-
   * centre current / lambda, in place; network() then multiplies by lambda
   * for amps. */
  for (int p = 0; p < np; p++) {
    cplx *x = rhs + (size_t)p * n;
    compute_current_coefficients(ctx, x);
    cplx *dst = s->basis + ((size_t)k * np + p) * n;
    for (int i = 0; i < n; i++)
      dst[i] = x[i] * wl;
    for (int q = 0; q < np; q++)
      s->y[((size_t)k * np + q) * np + p] = dst[s->port_seg[q] - 1]; /* I_q / V_p */
  }

  m = engine_problem(ctx, e0);
  if (m) {
    snprintf(err, errlen, "at %g MHz: %s", s->freqs[k], m);
    return -1;
  }
  return 0;
}

/* Currents for port voltages v at frequency k into ctx (crnt a/b/c and
 * surface_cur), and the input power 0.5 Re sum v_p conj(I_p) that network()
 * would report for the same EX set. */
static double superpose(SolutionObject *s, int k, const cplx *v, int want_somnec)
{
  context_t *ctx = s->ctx;
  int n = s->nseg;
  engine_set_frequency(s, s->freqs[k], want_somnec);
  cplx *x = ctx->crnt.surface_cur;
  memset(x, 0, sizeof(cplx) * (size_t)n);
  for (int p = 0; p < s->np; p++) {
    if (v[p] == 0)
      continue;
    const cplx *r = s->raw + ((size_t)k * s->np + p) * n;
    for (int i = 0; i < n; i++)
      x[i] += v[p] * r[i];
  }
  ctx->vsorc.num_qdsrcs_used = 0;
  compute_current_coefficients(ctx, x);
  double pin = 0.0;
  for (int p = 0; p < s->np; p++)
    pin += 0.5 * creal(v[p] * conj(x[s->port_seg[p] - 1] * ctx->geometry.wavelength));
  return pin;
}

/* ------------------------------------------------------------------------ */
/* argument parsing helpers */

/* Item `k` of `seq` as an int / finite double, the error naming
 * what[idx].field.  Every validation failure is a ValueError. */
static int item_int(PyObject *seq, Py_ssize_t k, const char *what, Py_ssize_t idx,
                    const char *field, int *out)
{
  PyObject *o = PySequence_GetItem(seq, k);
  if (!o)
    goto bad;
  PyObject *ix = PyNumber_Index(o);
  Py_DECREF(o);
  if (!ix)
    goto bad;
  long v = PyLong_AsLong(ix);
  Py_DECREF(ix);
  if (v == -1 && PyErr_Occurred())
    goto bad;
  if (v < INT32_MIN || v > INT32_MAX) {
    PyErr_Format(PyExc_ValueError, "%s[%zd].%s: %ld is out of range", what, idx, field, v);
    return -1;
  }
  *out = (int)v;
  return 0;
bad:
  PyErr_Clear();
  PyErr_Format(PyExc_ValueError, "%s[%zd].%s: expected an integer", what, idx, field);
  return -1;
}

static int item_double(PyObject *seq, Py_ssize_t k, const char *what, Py_ssize_t idx,
                       const char *field, double *out)
{
  PyObject *o = PySequence_GetItem(seq, k);
  double v = -1.0;
  if (o) {
    v = PyFloat_AsDouble(o);
    Py_DECREF(o);
  }
  if (!o || (v == -1.0 && PyErr_Occurred())) {
    PyErr_Clear();
    PyErr_Format(PyExc_ValueError, "%s[%zd].%s: expected a real number", what, idx, field);
    return -1;
  }
  if (!isfinite(v)) {
    PyErr_Format(PyExc_ValueError, "%s[%zd].%s: must be finite", what, idx, field);
    return -1;
  }
  *out = v;
  return 0;
}

/* `obj` as a sequence of fixed-length records; returns the record count or
 * -1.  `rec` receives a new reference to record `k` on each call of
 * record_at(). */
static Py_ssize_t records(PyObject *obj, const char *what)
{
  if (!PySequence_Check(obj) || PyUnicode_Check(obj) || PyBytes_Check(obj)) {
    PyErr_Format(PyExc_ValueError, "%s: expected a sequence of records", what);
    return -1;
  }
  return PySequence_Size(obj);
}

static PyObject *record_at(PyObject *seq, Py_ssize_t k, const char *what, Py_ssize_t len)
{
  PyObject *r = PySequence_GetItem(seq, k);
  if (!r)
    return NULL;
  if (!PySequence_Check(r) || PyUnicode_Check(r) || PySequence_Size(r) != len) {
    PyErr_Clear();
    PyErr_Format(PyExc_ValueError, "%s[%zd]: expected a sequence of %zd fields", what, k, len);
    Py_DECREF(r);
    return NULL;
  }
  return r;
}

static int point3(PyObject *rec, Py_ssize_t k, const char *what, Py_ssize_t idx,
                  const char *field, double out[3])
{
  static const char *axis[3] = {"x", "y", "z"};
  PyObject *o = PySequence_GetItem(rec, k);
  if (!o)
    return -1;
  if (!PySequence_Check(o) || PySequence_Size(o) != 3) {
    PyErr_Clear();
    Py_DECREF(o);
    PyErr_Format(PyExc_ValueError, "%s[%zd].%s: expected (x, y, z)", what, idx, field);
    return -1;
  }
  for (int a = 0; a < 3; a++) {
    char name[32];
    snprintf(name, sizeof(name), "%s.%s", field, axis[a]);
    if (item_double(o, a, what, idx, name, &out[a]) != 0) {
      Py_DECREF(o);
      return -1;
    }
  }
  Py_DECREF(o);
  return 0;
}

/* A C-contiguous buffer of `nitems` elements whose struct format is one of
 * `formats` (each element `itemsize` bytes).  nitems < 0 accepts any count.
 * On success the caller releases `view`. */
static int get_buffer(PyObject *obj, Py_buffer *view, const char *name,
                      const char *const *formats, Py_ssize_t itemsize, Py_ssize_t nitems)
{
  if (PyObject_GetBuffer(obj, view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) {
    PyErr_Clear();
    PyErr_Format(PyExc_ValueError, "%s: expected a C-contiguous buffer", name);
    return -1;
  }
  const char *f = view->format ? view->format : "B";
  if (*f == '<' || *f == '=' || *f == '@')
    f++;
  int fmt_ok = 0;
  for (const char *const *p = formats; *p; p++)
    if (strcmp(f, *p) == 0)
      fmt_ok = 1;
  if (!fmt_ok || view->itemsize != itemsize) {
    PyErr_Format(PyExc_ValueError, "%s: buffer format '%s' (itemsize %zd), expected '%s'",
                 name, view->format ? view->format : "B", view->itemsize, formats[0]);
    PyBuffer_Release(view);
    return -1;
  }
  Py_ssize_t count = view->len / itemsize;
  if (nitems >= 0 && count != nitems) {
    PyErr_Format(PyExc_ValueError, "%s: expected %zd values, got %zd", name, nitems, count);
    PyBuffer_Release(view);
    return -1;
  }
  return 0;
}

static const char *const F64[] = {"d", NULL};
static const char *const C128[] = {"Zd", NULL};

/* ------------------------------------------------------------------------ */
/* Structure */

/* Global 1-based segment for (tag, seg): tag 0 numbers segments absolutely,
 * as NEC's EX/LD cards do; 0 when there is no such segment. */
static int find_segment(const context_t *ctx, int tag, int seg)
{
  if (seg <= 0)
    return 0;
  if (tag == 0)
    return seg <= ctx->geometry.num_segs ? seg : 0;
  int cnt = 0;
  for (int i = 0; i < ctx->geometry.num_segs; i++)
    if (ctx->geometry.tag_nums[i] == tag && ++cnt == seg)
      return i + 1;
  return 0;
}

static int count_tag(const context_t *ctx, int tag)
{
  int cnt = 0;
  for (int i = 0; i < ctx->geometry.num_segs; i++)
    cnt += ctx->geometry.tag_nums[i] == tag;
  return cnt;
}

static int parse_wires(PyObject *obj, spec_t *s)
{
  Py_ssize_t n = records(obj, "wires");
  if (n < 0)
    return -1;
  if (n == 0) {
    PyErr_SetString(PyExc_ValueError, "wires: at least one wire is required");
    return -1;
  }
  if (n > INT32_MAX / 2) {
    PyErr_SetString(PyExc_ValueError, "wires: too many");
    return -1;
  }
  s->w = PyMem_Calloc((size_t)n, sizeof(wire_spec));
  if (!s->w) {
    PyErr_NoMemory();
    return -1;
  }
  s->nw = (int)n;
  long total = 0;
  for (Py_ssize_t k = 0; k < n; k++) {
    wire_spec *w = &s->w[k];
    PyObject *r = record_at(obj, k, "wires", 5);
    if (!r)
      return -1;
    int rc = item_int(r, 0, "wires", k, "tag", &w->tag) ||
             item_int(r, 1, "wires", k, "nseg", &w->nseg) ||
             point3(r, 2, "wires", k, "end1", w->e1) ||
             point3(r, 3, "wires", k, "end2", w->e2) ||
             item_double(r, 4, "wires", k, "radius", &w->radius);
    Py_DECREF(r);
    if (rc)
      return -1;
    if (w->tag < 0) {
      PyErr_Format(PyExc_ValueError, "wires[%zd].tag: must be >= 0, got %d", k, w->tag);
      return -1;
    }
    if (w->nseg < 1) {
      PyErr_Format(PyExc_ValueError, "wires[%zd].nseg: must be >= 1, got %d", k, w->nseg);
      return -1;
    }
    if (!(w->radius > 0.0)) {
      PyErr_Format(PyExc_ValueError, "wires[%zd].radius: must be > 0", k);
      return -1;
    }
    double dx = w->e2[0] - w->e1[0], dy = w->e2[1] - w->e1[1], dz = w->e2[2] - w->e1[2];
    if (dx * dx + dy * dy + dz * dz <= 0.0) {
      PyErr_Format(PyExc_ValueError, "wires[%zd].end2: coincides with end1", k);
      return -1;
    }
    total += w->nseg;
    if (total > INT32_MAX / 4) {
      PyErr_Format(PyExc_ValueError, "wires[%zd].nseg: total segment count overflows", k);
      return -1;
    }
  }
  return 0;
}

static int parse_ports(PyObject *obj, spec_t *s)
{
  Py_ssize_t n = records(obj, "ports");
  if (n < 0)
    return -1;
  if (n == 0) {
    PyErr_SetString(PyExc_ValueError, "ports: at least one port is required");
    return -1;
  }
  if (n > 4096) {
    PyErr_SetString(PyExc_ValueError, "ports: at most 4096");
    return -1;
  }
  s->p = PyMem_Calloc((size_t)n, sizeof(port_spec));
  if (!s->p) {
    PyErr_NoMemory();
    return -1;
  }
  s->np = (int)n;
  for (Py_ssize_t k = 0; k < n; k++) {
    PyObject *r = record_at(obj, k, "ports", 2);
    if (!r)
      return -1;
    int rc = item_int(r, 0, "ports", k, "tag", &s->p[k].tag) ||
             item_int(r, 1, "ports", k, "seg", &s->p[k].seg);
    Py_DECREF(r);
    if (rc)
      return -1;
  }
  return 0;
}

static int parse_loads(PyObject *obj, spec_t *s)
{
  static const char *fname[3] = {"f1", "f2", "f3"};
  if (obj == Py_None)
    return 0;
  Py_ssize_t n = records(obj, "loads");
  if (n < 0)
    return -1;
  if (n > 200) { /* add_loading()'s table limit */
    PyErr_SetString(PyExc_ValueError, "loads: at most 200");
    return -1;
  }
  s->l = PyMem_Calloc((size_t)(n ? n : 1), sizeof(load_spec));
  if (!s->l) {
    PyErr_NoMemory();
    return -1;
  }
  s->nl = (int)n;
  for (Py_ssize_t k = 0; k < n; k++) {
    load_spec *l = &s->l[k];
    PyObject *r = record_at(obj, k, "loads", 7);
    if (!r)
      return -1;
    int rc = item_int(r, 0, "loads", k, "type", &l->type) ||
             item_int(r, 1, "loads", k, "tag", &l->tag) ||
             item_int(r, 2, "loads", k, "seg_from", &l->from) ||
             item_int(r, 3, "loads", k, "seg_to", &l->to);
    for (int a = 0; !rc && a < 3; a++)
      rc = item_double(r, 4 + a, "loads", k, fname[a], &l->f[a]);
    Py_DECREF(r);
    if (rc)
      return -1;
    /* LD 0-5 only: 6 (trap) and 7 (insulation) are OpenNEC extensions with
     * their own parameters, and apply_impedance_loading() asserts above 7 */
    if (l->type < 0 || l->type > 5) {
      PyErr_Format(PyExc_ValueError, "loads[%zd].type: must be 0..5 (LD 0-5), got %d", k, l->type);
      return -1;
    }
    if (l->type == 5 && !(l->f[0] > 0.0)) {
      PyErr_Format(PyExc_ValueError, "loads[%zd].f1: wire conductivity must be > 0", k);
      return -1;
    }
  }
  return 0;
}

/* Reference checks that need the engine's segment numbering. */
static int check_refs(const spec_t *s, const context_t *ctx, int *port_seg)
{
  int nseg = ctx->geometry.num_segs;
  for (int k = 0; k < s->np; k++) {
    const port_spec *p = &s->p[k];
    int g = find_segment(ctx, p->tag, p->seg);
    if (g == 0) {
      if (p->seg <= 0)
        PyErr_Format(PyExc_ValueError, "ports[%d].seg: must be >= 1, got %d", k, p->seg);
      else if (p->tag != 0 && count_tag(ctx, p->tag) == 0)
        PyErr_Format(PyExc_ValueError, "ports[%d].tag: no wire has tag %d", k, p->tag);
      else
        PyErr_Format(PyExc_ValueError, "ports[%d].seg: no segment %d on tag %d", k, p->seg, p->tag);
      return -1;
    }
    for (int j = 0; j < k; j++)
      if (port_seg[j] == g) {
        PyErr_Format(PyExc_ValueError, "ports[%d]: same segment as ports[%d] (global segment %d)",
                     k, j, g);
        return -1;
      }
    port_seg[k] = g;
  }
  for (int k = 0; k < s->nl; k++) {
    const load_spec *l = &s->l[k];
    int to = l->to == 0 ? l->from : l->to;
    if (l->tag < 0) {
      PyErr_Format(PyExc_ValueError, "loads[%d].tag: must be >= 0, got %d", k, l->tag);
      return -1;
    }
    int avail = (l->tag == 0) ? nseg : count_tag(ctx, l->tag);
    if (l->tag != 0 && avail == 0) {
      PyErr_Format(PyExc_ValueError, "loads[%d].tag: no wire has tag %d", k, l->tag);
      return -1;
    }
    if (l->from == 0 && l->to == 0)
      continue; /* the whole tag, or with tag 0 every segment */
    if (l->from < 1 || l->from > avail) {
      PyErr_Format(PyExc_ValueError, "loads[%d].seg_from: %d is outside 1..%d", k, l->from, avail);
      return -1;
    }
    if (to < l->from || to > avail) {
      PyErr_Format(PyExc_ValueError, "loads[%d].seg_to: %d is outside %d..%d", k, to, l->from, avail);
      return -1;
    }
  }
  return 0;
}

static void Structure_dealloc(PyObject *op)
{
  StructureObject *self = (StructureObject *)op;
  PyTypeObject *tp = Py_TYPE(op);
  spec_free(&self->spec);
  PyMem_Free(self->port_seg);
  freefunc tp_free = (freefunc)PyType_GetSlot(tp, Py_tp_free);
  tp_free(op);
  Py_DECREF(tp);
}

static PyObject *Structure_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  static char *kwlist[] = {"wires", "ports", "loads", "ground", "epsr", "sigma",
                           "extended_kernel", NULL};
  PyObject *wires, *ports, *loads = Py_None;
  int ground = GROUND_FREE, ek = 0;
  double epsr = 1.0, sigma = 0.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O$iddp", kwlist, &wires, &ports, &loads,
                                   &ground, &epsr, &sigma, &ek))
    return NULL;

  allocfunc alloc = (allocfunc)PyType_GetSlot(type, Py_tp_alloc);
  StructureObject *self = (StructureObject *)alloc(type, 0);
  if (!self)
    return NULL;
  spec_t *s = &self->spec;
  s->ground = ground;
  s->epsr = epsr;
  s->sigma = sigma;
  s->ek = ek;

  if (ground != GROUND_FREE && ground != GROUND_REFLECTION && ground != GROUND_PERFECT &&
      ground != GROUND_SOMMERFELD) {
    PyErr_Format(PyExc_ValueError, "ground: must be -1 (free), 0 (reflection), 1 (perfect) "
                 "or 2 (Sommerfeld), got %d", ground);
    goto fail;
  }
  if (ground == GROUND_REFLECTION || ground == GROUND_SOMMERFELD) {
    if (!isfinite(epsr) || epsr < 1.0) {
      PyErr_SetString(PyExc_ValueError, "ground.epsr: must be finite and >= 1");
      goto fail;
    }
    if (!isfinite(sigma) || sigma < 0.0) {
      PyErr_SetString(PyExc_ValueError, "ground.sigma: must be finite and >= 0 S/m");
      goto fail;
    }
  }
  if (parse_wires(wires, s) || parse_ports(ports, s) || parse_loads(loads, s))
    goto fail;

  {
    char err[512];
    context_t *ctx = engine_build(s, err, sizeof(err));
    if (!ctx) {
      PyErr_SetString(PyExc_ValueError, err);
      goto fail;
    }
    self->nseg = ctx->geometry.num_segs;
    self->port_seg = PyMem_Calloc((size_t)s->np, sizeof(int));
    int rc = self->port_seg ? check_refs(s, ctx, self->port_seg) : (PyErr_NoMemory(), -1);
    engine_destroy(ctx);
    if (rc)
      goto fail;
  }
  return (PyObject *)self;

fail:
  Py_DECREF(self);
  return NULL;
}

/* 1 = cancel requested, 0 = carry on, -1 = the check raised.  `cancel` is a
 * threading.Event-like object (has is_set) or a zero-argument callable. */
static int cancel_requested(PyObject *cancel)
{
  PyObject *r;
  if (PyObject_HasAttrString(cancel, "is_set"))
    r = PyObject_CallMethod(cancel, "is_set", NULL);
  else
    r = PyObject_CallNoArgs(cancel);
  if (!r)
    return -1;
  int t = PyObject_IsTrue(r);
  Py_DECREF(r);
  return t;
}

static void Solution_dealloc(PyObject *op);

static PyObject *Structure_solve(PyObject *op, PyObject *args, PyObject *kwds)
{
  static char *kwlist[] = {"freqs_mhz", "cancel", "progress", NULL};
  StructureObject *st = (StructureObject *)op;
  PyObject *freqs_obj, *cancel = Py_None, *progress = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$OO", kwlist, &freqs_obj, &cancel, &progress))
    return NULL;
  if (cancel != Py_None && !PyObject_HasAttrString(cancel, "is_set") && !PyCallable_Check(cancel)) {
    PyErr_SetString(PyExc_ValueError, "cancel: expected a threading.Event or a callable");
    return NULL;
  }
  if (progress != Py_None && !PyCallable_Check(progress)) {
    PyErr_SetString(PyExc_ValueError, "progress: expected a callable(done, total)");
    return NULL;
  }

  Py_buffer fv;
  if (get_buffer(freqs_obj, &fv, "freqs_mhz", F64, sizeof(double), -1) != 0)
    return NULL;
  Py_ssize_t nf = fv.len / (Py_ssize_t)sizeof(double);
  if (nf < 1 || nf > INT32_MAX) {
    PyBuffer_Release(&fv);
    PyErr_SetString(PyExc_ValueError, "freqs_mhz: at least one frequency is required");
    return NULL;
  }
  for (Py_ssize_t k = 0; k < nf; k++) {
    double f = ((const double *)fv.buf)[k];
    if (!isfinite(f) || f <= 0.0) {
      PyBuffer_Release(&fv);
      PyErr_Format(PyExc_ValueError, "freqs_mhz[%zd]: must be finite and > 0", k);
      return NULL;
    }
  }

  PyTypeObject *stype = (PyTypeObject *)SolutionType;
  allocfunc alloc = (allocfunc)PyType_GetSlot(stype, Py_tp_alloc);
  SolutionObject *s = (SolutionObject *)alloc(stype, 0);
  cplx *cm = NULL, *rhs = NULL;
  int *ip = NULL;
  if (!s) {
    PyBuffer_Release(&fv);
    return NULL;
  }
  s->nf = (int)nf;
  s->np = st->spec.np;
  s->nseg = st->nseg;
  size_t n = (size_t)s->nseg, np = (size_t)s->np;
  s->freqs = PyMem_Malloc(sizeof(double) * (size_t)nf);
  if (s->freqs)
    memcpy(s->freqs, fv.buf, sizeof(double) * (size_t)nf);
  PyBuffer_Release(&fv);
  s->lock = PyThread_allocate_lock();
  s->port_seg = PyMem_Malloc(sizeof(int) * np);
  s->mx = PyMem_Malloc(sizeof(double) * n);
  s->my = PyMem_Malloc(sizeof(double) * n);
  s->mz = PyMem_Malloc(sizeof(double) * n);
  s->mlen = PyMem_Malloc(sizeof(double) * n);
  s->mrad = PyMem_Malloc(sizeof(double) * n);
  s->raw = PyMem_Malloc(sizeof(cplx) * (size_t)nf * np * n);
  s->basis = PyMem_Malloc(sizeof(cplx) * (size_t)nf * np * n);
  s->y = PyMem_Malloc(sizeof(cplx) * (size_t)nf * np * np);
  if (!s->freqs || !s->lock || !s->port_seg || !s->mx || !s->my || !s->mz || !s->mlen ||
      !s->mrad || !s->raw || !s->basis || !s->y || spec_copy(&s->spec, &st->spec) != 0) {
    PyErr_NoMemory();
    goto fail;
  }
  memcpy(s->port_seg, st->port_seg, sizeof(int) * np);

  /* The interaction matrix at the engine's own size, N(N+2) (reporting.c's
   * iresrv; storage case 1 touches only the N x N block), allocated once and
   * refilled for every frequency.  Plain malloc: the engine's allocator, not
   * Python's, owns arrays the engine might free. */
  if (n > (SIZE_MAX / sizeof(cplx)) / (n + 2)) {
    PyErr_NoMemory();
    goto fail;
  }
  cm = malloc(sizeof(cplx) * n * (n + 2));
  rhs = malloc(sizeof(cplx) * n * np);
  ip = malloc(sizeof(int) * n);
  if (!cm || !rhs || !ip) {
    PyErr_NoMemory();
    goto fail;
  }

  char err[512];
  err[0] = '\0';
  int rc;
  Py_BEGIN_ALLOW_THREADS
  s->ctx = engine_build(&s->spec, err, sizeof(err));
  rc = s->ctx ? 0 : -1;
  if (rc == 0) {
    geometry_t *g = &s->ctx->geometry;
    memcpy(s->mx, g->x_center, sizeof(double) * n);
    memcpy(s->my, g->y_center, sizeof(double) * n);
    memcpy(s->mz, g->z_center, sizeof(double) * n);
    memcpy(s->mlen, g->half_len, sizeof(double) * n);
    memcpy(s->mrad, g->radius, sizeof(double) * n);
    /* storage case 1 (whole matrix in core, one symmetry section) */
    if (factor_block_matrix(s->ctx, (int)n, (int)n, (int)(n * (n + 2) > INT32_MAX ? INT32_MAX : n * (n + 2)),
                            g->symmetry_flag) != 0 || s->ctx->matpar.storage_case != 1) {
      snprintf(err, sizeof(err), "matrix layout: expected one in-core block");
      rc = -1;
    }
  }
  Py_END_ALLOW_THREADS
  if (rc != 0) {
    PyErr_SetString(PortyError, err);
    goto fail;
  }

  for (int k = 0; k < s->nf; k++) {
    if (cancel != Py_None) {
      int c = cancel_requested(cancel);
      if (c < 0)
        goto fail;
      if (c) {
        PyErr_Format(Cancelled, "solve cancelled before frequency %d of %d", k + 1, s->nf);
        goto fail;
      }
    }
    Py_BEGIN_ALLOW_THREADS
    rc = solve_frequency(s, k, cm, ip, rhs, err, sizeof(err));
    Py_END_ALLOW_THREADS
    if (rc != 0) {
      PyErr_SetString(PortyError, err);
      goto fail;
    }
    if (PyErr_CheckSignals() != 0)
      goto fail;
    if (progress != Py_None) {
      PyObject *r = PyObject_CallFunction(progress, "ii", k + 1, s->nf);
      if (!r)
        goto fail;
      Py_DECREF(r);
    }
  }
  free(cm);
  free(rhs);
  free(ip);
  return (PyObject *)s;

fail:
  free(cm);
  free(rhs);
  free(ip);
  Py_DECREF(s); /* Solution_dealloc frees whatever was allocated */
  return NULL;
}

static PyObject *Structure_get_nseg(PyObject *op, void *closure)
{
  (void)closure;
  return PyLong_FromLong(((StructureObject *)op)->nseg);
}

static PyObject *Structure_get_nports(PyObject *op, void *closure)
{
  (void)closure;
  return PyLong_FromLong(((StructureObject *)op)->spec.np);
}

static PyMethodDef Structure_methods[] = {
  {"solve", (PyCFunction)(void (*)(void))Structure_solve, METH_VARARGS | METH_KEYWORDS,
   "solve(freqs_mhz, *, cancel=None, progress=None) -> Solution\n\n"
   "freqs_mhz: C-contiguous float64 buffer.  The GIL is released while the\n"
   "engine fills, factors and solves; between frequencies cancel (an Event or\n"
   "a callable returning bool) is checked and progress(done, total) called."},
  {NULL, NULL, 0, NULL}
};

static PyGetSetDef Structure_getset[] = {
  {"nseg", Structure_get_nseg, NULL, "number of wire segments", NULL},
  {"nports", Structure_get_nports, NULL, "number of ports", NULL},
  {NULL, NULL, NULL, NULL, NULL}
};

static PyType_Slot Structure_slots[] = {
  {Py_tp_new, Structure_new},
  {Py_tp_dealloc, Structure_dealloc},
  {Py_tp_methods, Structure_methods},
  {Py_tp_getset, Structure_getset},
  {Py_tp_doc, "Structure(wires, ports, loads=None, *, ground=-1, epsr=1, sigma=0, "
              "extended_kernel=False)\n\n"
              "wires: (tag, nseg, (x1,y1,z1), (x2,y2,z2), radius) in metres.\n"
              "ports: (tag, seg); tag 0 numbers segments absolutely.\n"
              "loads: (type 0-5, tag, seg_from, seg_to, f1, f2, f3), LD card fields."},
  {0, NULL}
};

static PyType_Spec Structure_spec = {
  "pyopennec._porty.Structure", sizeof(StructureObject), 0, Py_TPFLAGS_DEFAULT, Structure_slots
};

/* ------------------------------------------------------------------------ */
/* Solution */

static void Solution_dealloc(PyObject *op)
{
  SolutionObject *s = (SolutionObject *)op;
  PyTypeObject *tp = Py_TYPE(op);
  engine_destroy(s->ctx);
  if (s->lock)
    PyThread_free_lock(s->lock);
  spec_free(&s->spec);
  PyMem_Free(s->freqs);
  PyMem_Free(s->port_seg);
  PyMem_Free(s->mx);
  PyMem_Free(s->my);
  PyMem_Free(s->mz);
  PyMem_Free(s->mlen);
  PyMem_Free(s->mrad);
  PyMem_Free(s->raw);
  PyMem_Free(s->basis);
  PyMem_Free(s->y);
  freefunc tp_free = (freefunc)PyType_GetSlot(tp, Py_tp_free);
  tp_free(op);
  Py_DECREF(tp);
}

static PyObject *bytes_of(const void *p, size_t nbytes)
{
  return PyBytes_FromStringAndSize((const char *)p, (Py_ssize_t)nbytes);
}

#define S(op) ((SolutionObject *)(op))

static PyObject *Sol_nf(PyObject *op, void *c) { (void)c; return PyLong_FromLong(S(op)->nf); }
static PyObject *Sol_np(PyObject *op, void *c) { (void)c; return PyLong_FromLong(S(op)->np); }
static PyObject *Sol_nseg(PyObject *op, void *c) { (void)c; return PyLong_FromLong(S(op)->nseg); }
static PyObject *Sol_freqs(PyObject *op, void *c)
{
  (void)c;
  return bytes_of(S(op)->freqs, sizeof(double) * (size_t)S(op)->nf);
}
static PyObject *Sol_y(PyObject *op, void *c)
{
  (void)c;
  SolutionObject *s = S(op);
  return bytes_of(s->y, sizeof(cplx) * (size_t)s->nf * s->np * s->np);
}
static PyObject *Sol_basis(PyObject *op, void *c)
{
  (void)c;
  SolutionObject *s = S(op);
  return bytes_of(s->basis, sizeof(cplx) * (size_t)s->nf * s->np * s->nseg);
}
static PyObject *Sol_port_segments(PyObject *op, void *c)
{
  (void)c;
  SolutionObject *s = S(op);
  int32_t *t = PyMem_Malloc(sizeof(int32_t) * (size_t)s->np);
  if (!t)
    return PyErr_NoMemory();
  for (int k = 0; k < s->np; k++)
    t[k] = s->port_seg[k];
  PyObject *r = bytes_of(t, sizeof(int32_t) * (size_t)s->np);
  PyMem_Free(t);
  return r;
}

/* (centres (n,3) float64 m, lengths m, radii m, tags int32, seg-in-tag int32) */
static PyObject *Sol_segments(PyObject *op, void *c)
{
  (void)c;
  SolutionObject *s = S(op);
  size_t n = (size_t)s->nseg;
  double *xyz = PyMem_Malloc(sizeof(double) * 3 * n);
  int32_t *tag = PyMem_Malloc(sizeof(int32_t) * n);
  int32_t *num = PyMem_Malloc(sizeof(int32_t) * n);
  PyObject *r = NULL;
  if (!xyz || !tag || !num) {
    PyErr_NoMemory();
    goto out;
  }
  for (size_t i = 0; i < n; i++) {
    xyz[3 * i] = s->mx[i];
    xyz[3 * i + 1] = s->my[i];
    xyz[3 * i + 2] = s->mz[i];
  }
  /* GW cards generate their segments in card order, so a segment's number
   * within its tag is the count of that tag's segments on earlier wires plus
   * its index on this one (find_segment()'s numbering); tag 0 is absolute. */
  size_t i = 0;
  for (int k = 0; k < s->spec.nw; k++) {
    const wire_spec *w = &s->spec.w[k];
    int base = 0;
    for (int j = 0; j < k; j++)
      if (s->spec.w[j].tag == w->tag)
        base += s->spec.w[j].nseg;
    for (int m = 1; m <= w->nseg; m++, i++) {
      tag[i] = w->tag;
      num[i] = w->tag == 0 ? (int32_t)(i + 1) : base + m;
    }
  }
  r = Py_BuildValue("(NNNNN)", bytes_of(xyz, sizeof(double) * 3 * n),
                    bytes_of(s->mlen, sizeof(double) * n), bytes_of(s->mrad, sizeof(double) * n),
                    bytes_of(tag, sizeof(int32_t) * n), bytes_of(num, sizeof(int32_t) * n));
out:
  PyMem_Free(xyz);
  PyMem_Free(tag);
  PyMem_Free(num);
  return r;
}

/* Frequency index and port-voltage vector shared by both field queries. */
static int field_args(SolutionObject *s, int i_f, PyObject *vobj, Py_buffer *vv)
{
  if (i_f < 0 || i_f >= s->nf) {
    PyErr_Format(PyExc_ValueError, "i_f: %d is outside 0..%d", i_f, s->nf - 1);
    return -1;
  }
  if (get_buffer(vobj, vv, "port_voltages", C128, sizeof(cplx), s->np) != 0)
    return -1;
  const cplx *v = vv->buf;
  for (int p = 0; p < s->np; p++)
    if (!isfinite(creal(v[p])) || !isfinite(cimag(v[p]))) {
      PyBuffer_Release(vv);
      PyErr_Format(PyExc_ValueError, "port_voltages[%d]: must be finite", p);
      return -1;
    }
  return 0;
}

/* _far_field(i_f, v, theta_deg, phi_deg) ->
 *   (gains bytes (n,3) float64 dBi [total, vertical, horizontal],
 *    fields bytes (n,2) complex128 [E_theta, E_phi] (r E, volts),
 *    valid bytes (n,) uint8, power_in float W)
 * Power gain, normalised to the input power the port voltages deliver, as
 * an RP card with XNDA 0 reports it. */
static PyObject *Sol_far_field(PyObject *op, PyObject *args)
{
  SolutionObject *s = S(op);
  int i_f;
  PyObject *vobj, *tobj, *pobj;
  if (!PyArg_ParseTuple(args, "iOOO", &i_f, &vobj, &tobj, &pobj))
    return NULL;
  Py_buffer vv, tv, pv;
  if (field_args(s, i_f, vobj, &vv) != 0)
    return NULL;
  if (get_buffer(tobj, &tv, "theta_deg", F64, sizeof(double), -1) != 0) {
    PyBuffer_Release(&vv);
    return NULL;
  }
  Py_ssize_t npt = tv.len / (Py_ssize_t)sizeof(double);
  if (get_buffer(pobj, &pv, "phi_deg", F64, sizeof(double), npt) != 0) {
    PyBuffer_Release(&vv);
    PyBuffer_Release(&tv);
    return NULL;
  }
  const double *th = tv.buf, *ph = pv.buf;
  for (Py_ssize_t k = 0; k < npt; k++)
    if (!isfinite(th[k]) || !isfinite(ph[k])) {
      PyErr_Format(PyExc_ValueError, "theta_deg/phi_deg[%zd]: must be finite", k);
      goto fail_buffers;
    }

  size_t un = (size_t)npt;
  double *gains = PyMem_Malloc(sizeof(double) * 3 * (un ? un : 1));
  cplx *fields = PyMem_Malloc(sizeof(cplx) * 2 * (un ? un : 1));
  uint8_t *valid = PyMem_Malloc(un ? un : 1);
  if (!gains || !fields || !valid) {
    PyMem_Free(gains);
    PyMem_Free(fields);
    PyMem_Free(valid);
    PyErr_NoMemory();
    goto fail_buffers;
  }

  double pin;
  int bad_points = 0;
  Py_BEGIN_ALLOW_THREADS
  PyThread_acquire_lock(s->lock, WAIT_LOCK);
  context_t *ctx = s->ctx;
  pin = superpose(s, i_f, vv.buf, 0);
  if (pin > 0.0) {
    field_pattern_t *fp = &ctx->fpat;
    fp->num_theta = fp->num_phi = 1;
    fp->theta_step = fp->phi_step = 0.0;
    fp->range = 0.0;
    fp->norm_gain = 0.0;
    fp->gain_type = fp->avg_power_flag = fp->normalize_gain = fp->pol_axis = 0;
    fp->excitation_type = 0;
    fp->power_in = pin;
    fp->network_loss = 0.0;
    fp->ohmic_loss = 0.0; /* enters directive gain only (gain_type 1) */
    ctx->gnd.far_field_type = 0;
    for (size_t k = 0; k < un; k++) {
      /* one point per call: the pattern loop starts at start - step and
       * adds step, so a zero step lands on the angle exactly */
      fp->theta_start = th[k];
      fp->phi_start = ph[k];
      compute_radiation_pattern(ctx);
      if (ctx->rpat.num_points == 1 && ctx->rpat.points) {
        const rpat_point_t *q = &ctx->rpat.points[0];
        gains[3 * k] = q->gtot;
        gains[3 * k + 1] = q->gnv;
        gains[3 * k + 2] = q->gnh;
        fields[2 * k] = q->ethm * cexp(I * q->etha * TA);
        fields[2 * k + 1] = q->ephm * cexp(I * q->epha * TA);
        valid[k] = 1;
      } else {
        /* below the ground horizon: NEC-2 computes nothing there */
        gains[3 * k] = gains[3 * k + 1] = gains[3 * k + 2] = -999.99;
        fields[2 * k] = fields[2 * k + 1] = 0.0;
        valid[k] = 0;
        bad_points += !(ctx->gnd.has_ground == 2 && th[k] > 90.01);
      }
      free(ctx->rpat.points);
      ctx->rpat.points = NULL;
      ctx->rpat.num_points = 0;
    }
  }
  PyThread_release_lock(s->lock);
  Py_END_ALLOW_THREADS

  PyObject *r = NULL;
  if (!(pin > 0.0)) {
    char msg[128];
    snprintf(msg, sizeof(msg), "port_voltages: deliver no input power (P_in = %g W)", pin);
    PyErr_SetString(PyExc_ValueError, msg);
  }
  else if (bad_points)
    PyErr_Format(PortyError, "radiation pattern returned no value at %d points", bad_points);
  else
    r = Py_BuildValue("(NNNd)", bytes_of(gains, sizeof(double) * 3 * un),
                      bytes_of(fields, sizeof(cplx) * 2 * un), bytes_of(valid, un), pin);
  PyMem_Free(gains);
  PyMem_Free(fields);
  PyMem_Free(valid);
  PyBuffer_Release(&vv);
  PyBuffer_Release(&tv);
  PyBuffer_Release(&pv);
  return r;

fail_buffers:
  PyBuffer_Release(&vv);
  PyBuffer_Release(&tv);
  PyBuffer_Release(&pv);
  return NULL;
}

/* _near_field(i_f, v, points, kind) -> bytes (n,3) complex128: E (V/m) for
 * kind 0, H (A/m) for kind 1, at points given in metres (n*3 float64). */
static PyObject *Sol_near_field(PyObject *op, PyObject *args)
{
  SolutionObject *s = S(op);
  int i_f, kind;
  PyObject *vobj, *xobj;
  if (!PyArg_ParseTuple(args, "iOOi", &i_f, &vobj, &xobj, &kind))
    return NULL;
  if (kind != 0 && kind != 1) {
    PyErr_SetString(PyExc_ValueError, "kind: 0 (E) or 1 (H)");
    return NULL;
  }
  Py_buffer vv, xv;
  if (field_args(s, i_f, vobj, &vv) != 0)
    return NULL;
  if (get_buffer(xobj, &xv, "points", F64, sizeof(double), -1) != 0) {
    PyBuffer_Release(&vv);
    return NULL;
  }
  Py_ssize_t ncoord = xv.len / (Py_ssize_t)sizeof(double);
  const double *xyz = xv.buf;
  if (ncoord % 3 != 0) {
    PyErr_SetString(PyExc_ValueError, "points: expected (n, 3) metres");
    goto fail;
  }
  for (Py_ssize_t k = 0; k < ncoord; k++)
    if (!isfinite(xyz[k])) {
      PyErr_Format(PyExc_ValueError, "points[%zd]: must be finite", k / 3);
      goto fail;
    }
  size_t npt = (size_t)ncoord / 3;
  cplx *out = PyMem_Malloc(sizeof(cplx) * 3 * (npt ? npt : 1));
  if (!out) {
    PyErr_NoMemory();
    goto fail;
  }
  Py_BEGIN_ALLOW_THREADS
  PyThread_acquire_lock(s->lock, WAIT_LOCK);
  context_t *ctx = s->ctx;
  superpose(s, i_f, vv.buf, 1);
  double wl = ctx->geometry.wavelength;
  for (size_t k = 0; k < npt; k++) {
    cplx ex, ey, ez;
    /* compute_near_field()'s own scaling: metres / lambda */
    double x = xyz[3 * k] / wl, y = xyz[3 * k + 1] / wl, z = xyz[3 * k + 2] / wl;
    if (kind == 0)
      near_e_field(ctx, x, y, z, &ex, &ey, &ez);
    else
      near_h_field(ctx, x, y, z, &ex, &ey, &ez);
    out[3 * k] = ex;
    out[3 * k + 1] = ey;
    out[3 * k + 2] = ez;
  }
  PyThread_release_lock(s->lock);
  Py_END_ALLOW_THREADS
  PyObject *r = bytes_of(out, sizeof(cplx) * 3 * npt);
  PyMem_Free(out);
  PyBuffer_Release(&vv);
  PyBuffer_Release(&xv);
  return r;

fail:
  PyBuffer_Release(&vv);
  PyBuffer_Release(&xv);
  return NULL;
}

static PyMethodDef Solution_methods[] = {
  {"_far_field", Sol_far_field, METH_VARARGS,
   "_far_field(i_f, port_voltages, theta_deg, phi_deg) -> (gains, fields, valid, power_in)"},
  {"_near_field", Sol_near_field, METH_VARARGS,
   "_near_field(i_f, port_voltages, points, kind) -> fields"},
  {NULL, NULL, 0, NULL}
};

static PyGetSetDef Solution_getset[] = {
  {"nf", Sol_nf, NULL, "number of frequencies", NULL},
  {"nports", Sol_np, NULL, "number of ports", NULL},
  {"nseg", Sol_nseg, NULL, "number of segments", NULL},
  {"freqs_mhz", Sol_freqs, NULL, "bytes: (nf,) float64", NULL},
  {"y", Sol_y, NULL, "bytes: (nf, P, P) complex128 port admittance, S", NULL},
  {"basis_currents", Sol_basis, NULL, "bytes: (nf, P, nseg) complex128, A per V", NULL},
  {"port_segments", Sol_port_segments, NULL, "bytes: (P,) int32 global 1-based segments", NULL},
  {"segments", Sol_segments, NULL,
   "(centres (nseg,3) f64 m, lengths f64 m, radii f64 m, tags i32, seg-in-tag i32), as bytes",
   NULL},
  {NULL, NULL, NULL, NULL, NULL}
};

static PyType_Slot Solution_slots[] = {
  {Py_tp_dealloc, Solution_dealloc},
  {Py_tp_methods, Solution_methods},
  {Py_tp_getset, Solution_getset},
  {Py_tp_doc, "A solved Structure: per-frequency port admittances and basis currents."},
  {0, NULL}
};

static PyType_Spec Solution_spec = {
  "pyopennec._porty.Solution", sizeof(SolutionObject), 0,
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_DISALLOW_INSTANTIATION, Solution_slots
};

/* ------------------------------------------------------------------------ */
/* module */

static struct PyModuleDef porty_module = {
  PyModuleDef_HEAD_INIT, "pyopennec._porty",
  "OpenNEC behind an N-port admittance interface (raw CPython C API).", -1,
  NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__porty(void)
{
  PyObject *m = PyModule_Create(&porty_module);
  if (!m)
    return NULL;
  StructureType = PyType_FromSpec(&Structure_spec);
  SolutionType = PyType_FromSpec(&Solution_spec);
  PortyError = PyErr_NewException("pyopennec._porty.PortyError", PyExc_RuntimeError, NULL);
  Cancelled = PyErr_NewException("pyopennec._porty.Cancelled", PyExc_Exception, NULL);
  if (!StructureType || !SolutionType || !PortyError || !Cancelled ||
      PyModule_AddObjectRef(m, "Structure", StructureType) < 0 ||
      PyModule_AddObjectRef(m, "Solution", SolutionType) < 0 ||
      PyModule_AddObjectRef(m, "PortyError", PortyError) < 0 ||
      PyModule_AddObjectRef(m, "Cancelled", Cancelled) < 0 ||
#ifdef Py_LIMITED_API
      PyModule_AddIntConstant(m, "LIMITED_API", Py_LIMITED_API) < 0 ||
#else
      PyModule_AddIntConstant(m, "LIMITED_API", 0) < 0 ||
#endif
      PyModule_AddIntConstant(m, "LAPACK_DIRECT", PORTY_LAPACK) < 0 ||
      PyModule_AddStringConstant(m, "OPENNEC_VERSION", VERSION_STRING) < 0 ||
      PyModule_AddStringConstant(m, "BACKEND", PYOPENNEC_BACKEND) < 0 ||
      PyModule_AddObjectRef(m, "OPENMP", PYOPENNEC_OPENMP ? Py_True : Py_False) < 0) {
    Py_DECREF(m);
    return NULL;
  }
  return m;
}
