# =============================================================================
# fEGarch output-fixture generator  (quantica clean-room port — Phase 0)
# =============================================================================
#
# CLEAN-ROOM BOUNDARY (CLAUDE.md §12): this script *runs* the fEGarch package and
# records the NUMERIC OUTPUT of its PUBLIC functions only. It does NOT read, open,
# or inspect fEGarch's source code. Capturing output for validation fixtures is the
# permitted method (constraint 2); reading internals is forbidden (constraint 1).
#
# The committed CSV/JSON fixtures let the R-free Python test suite check the
# clean-room reimplementation against fEGarch's own output. R is generated ONCE and
# is NEVER a CI dependency.
#
# Reproduce:  Rscript tests/fixtures/fegarch/generate_fixtures.R
# Requires:   R with the fEGarch package installed (install.packages("fEGarch")).
# =============================================================================

suppressMessages(library(fEGarch))

# --- locate this script's directory so outputs land beside it -----------------
.args <- commandArgs(FALSE)
.file <- sub("^--file=", "", .args[grep("^--file=", .args)])
OUTDIR <- if (length(.file)) normalizePath(dirname(.file)) else getwd()
cat("writing fixtures to:", OUTDIR, "\n")

FEGARCH_VERSION <- as.character(packageVersion("fEGarch"))
R_VERSION <- R.version.string

# --- a minimal JSON writer (avoids a jsonlite dependency) ---------------------
.json_scalar <- function(x) {
  if (is.null(x)) return("null")
  if (is.logical(x)) return(if (x) "true" else "false")
  if (is.numeric(x)) return(trimws(formatC(x, format = "g", digits = 17)))
  paste0("\"", gsub("\"", "\\\\\"", as.character(x)), "\"")
}
.to_json <- function(obj, indent = 0) {
  pad <- strrep("  ", indent + 1L)
  close_pad <- strrep("  ", indent)
  if (is.list(obj)) {
    if (length(obj) == 0) return("{}")
    nms <- names(obj)
    parts <- vapply(seq_along(obj), function(i) {
      paste0(pad, "\"", nms[i], "\": ", .to_json(obj[[i]], indent + 1L))
    }, character(1))
    return(paste0("{\n", paste(parts, collapse = ",\n"), "\n", close_pad, "}"))
  }
  if (length(obj) != 1L) {  # numeric/character vector -> JSON array
    return(paste0("[", paste(vapply(obj, .json_scalar, character(1)), collapse = ", "), "]"))
  }
  .json_scalar(obj)
}
write_json <- function(obj, path) writeLines(.to_json(obj), path)

# high-precision numeric vector -> single-column CSV
write_series_csv <- function(x, path, header) {
  writeLines(c(header, sprintf("%.17g", as.numeric(x))), path)
}

# =============================================================================
# 1. Conditional-distribution fixtures (sampler-based).
#
# fEGarch exposes the standardized samplers r{norm,std,ged,ald,snorm,sstd,sged,
# sald}_s but NO public density/CDF/quantile functions, so the distribution layer
# is characterised by the OUTPUT of the samplers: empirical moments (confirming the
# mean-0 / variance-1 standardization) and empirical quantiles on a fixed grid
# (confirming the parameterization). The Python tests compare their analytic values
# within Monte-Carlo tolerance.
# =============================================================================

N_SAMPLES <- 1000000L
BASE_SEED <- 20240601L
PROBS <- c(0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995)

# (label, distribution code, sampler thunk given n).  `df`, `shape`, `P`, `skew`
# are fEGarch's own public argument names.
dist_specs <- list(
  list(label = "norm",              dist = "norm",  args = list(),                        draw = function(n) rnorm_s(n)),
  list(label = "std_df5",           dist = "std",   args = list(df = 5),                  draw = function(n) rstd_s(n, df = 5)),
  list(label = "std_df8",           dist = "std",   args = list(df = 8),                  draw = function(n) rstd_s(n, df = 8)),
  list(label = "ged_shape1.0",      dist = "ged",   args = list(shape = 1.0),            draw = function(n) rged_s(n, shape = 1.0)),
  list(label = "ged_shape1.5",      dist = "ged",   args = list(shape = 1.5),            draw = function(n) rged_s(n, shape = 1.5)),
  list(label = "ged_shape2.0",      dist = "ged",   args = list(shape = 2.0),            draw = function(n) rged_s(n, shape = 2.0)),
  list(label = "ald_P8",            dist = "ald",   args = list(P = 8),                  draw = function(n) rald_s(n, P = 8)),
  list(label = "ald_P2",            dist = "ald",   args = list(P = 2),                  draw = function(n) rald_s(n, P = 2)),
  list(label = "snorm_skew0.8",     dist = "snorm", args = list(skew = 0.8),             draw = function(n) rsnorm_s(n, skew = 0.8)),
  list(label = "snorm_skew1.3",     dist = "snorm", args = list(skew = 1.3),             draw = function(n) rsnorm_s(n, skew = 1.3)),
  list(label = "sstd_df6_skew0.85", dist = "sstd",  args = list(df = 6, skew = 0.85),    draw = function(n) rsstd_s(n, df = 6, skew = 0.85)),
  list(label = "sged_shape1.3_skew1.2", dist = "sged", args = list(shape = 1.3, skew = 1.2), draw = function(n) rsged_s(n, shape = 1.3, skew = 1.2)),
  list(label = "sald_P8_skew0.8",   dist = "sald",  args = list(P = 8, skew = 0.8),      draw = function(n) rsald_s(n, P = 8, skew = 0.8))
)

moments_rows <- list()
quantile_rows <- list()
for (i in seq_along(dist_specs)) {
  spec <- dist_specs[[i]]
  seed <- BASE_SEED + i
  set.seed(seed)
  x <- as.numeric(spec$draw(N_SAMPLES))
  m <- mean(x); s <- sd(x)
  skewness <- mean((x - m)^3) / s^3
  kurtosis <- mean((x - m)^4) / s^4          # raw (not excess) kurtosis; normal ~ 3
  q <- as.numeric(quantile(x, probs = PROBS, names = FALSE, type = 7))
  arg_str <- if (length(spec$args)) paste(names(spec$args), unlist(spec$args), sep = "=", collapse = ";") else ""
  moments_rows[[length(moments_rows) + 1L]] <- sprintf(
    "%s,%s,%s,%d,%d,%.17g,%.17g,%.17g,%.17g",
    spec$label, spec$dist, arg_str, N_SAMPLES, seed, m, var(x), skewness, kurtosis)
  for (j in seq_along(PROBS)) {
    quantile_rows[[length(quantile_rows) + 1L]] <- sprintf(
      "%s,%s,%.17g,%.17g", spec$label, spec$dist, PROBS[j], q[j])
  }
  cat(sprintf("  dist %-22s mean=%+.4f var=%.4f skew=%+.3f kurt=%.3f\n",
              spec$label, m, var(x), skewness, kurtosis))
}
writeLines(c("label,dist,args,n_samples,seed,mean,variance,skewness,kurtosis_raw",
             unlist(moments_rows)), file.path(OUTDIR, "distribution_moments.csv"))
writeLines(c("label,dist,prob,quantile", unlist(quantile_rows)),
           file.path(OUTDIR, "distribution_quantiles.csv"))

# =============================================================================
# 2. Model-fit fixtures on a COMMITTED SYNTHETIC series.
#
# We do NOT redistribute fEGarch's bundled SP500 data (its compilation is GPL-3).
# Instead a deterministic synthetic GARCH(1,1) series is generated here, committed
# as synthetic_returns.csv, and used as the fit input — so the Python tests are
# fully reproducible and R-free, reusing the exact committed input.
# =============================================================================

SIM_SEED <- 20240715L
SIM_N <- 2500L
SIM <- list(mu = 3e-4, omega = 2.88e-6, alpha = 0.08, beta = 0.90)  # ~1.2% daily sd
set.seed(SIM_SEED)
sig2 <- numeric(SIM_N); eps <- numeric(SIM_N)
sig2[1] <- SIM$omega / (1 - SIM$alpha - SIM$beta)
z <- rnorm(SIM_N)
for (t in seq_len(SIM_N)) {
  if (t > 1) sig2[t] <- SIM$omega + SIM$alpha * eps[t - 1]^2 + SIM$beta * sig2[t - 1]
  eps[t] <- sqrt(sig2[t]) * z[t]
}
returns <- SIM$mu + eps
write_series_csv(returns, file.path(OUTDIR, "synthetic_returns.csv"), "return")
cat(sprintf("  synthetic series: n=%d mean=%.2e sd=%.4f\n", SIM_N, mean(returns), sd(returns)))

fit_and_dump <- function(fit, name, model, cond_dist, trunc = "none") {
  # `trunc` records the truncation policy metadata: "none" for the short-memory models (default),
  # and the long-memory default L = n-1 (WP171 App. C.3) for the fractionally-integrated ones.
  p <- pars(fit)
  ic <- inf_criteria(fit)
  params <- as.list(as.numeric(p)); names(params) <- names(p)
  meta <- list(
    model = model, cond_dist = cond_dist, orders = c(1L, 1L),
    presample = 50L, trunc = trunc, mean_included = TRUE,
    n_obs = length(returns), input = "synthetic_returns.csv",
    parallel = FALSE, seed = SIM_SEED,
    params = params,
    loglikelihood = as.numeric(llhood(fit)),
    aic = as.numeric(ic[["aic"]]), bic = as.numeric(ic[["bic"]]),
    fegarch_version = FEGARCH_VERSION, r_version = R_VERSION)
  write_json(meta, file.path(OUTDIR, sprintf("fit_%s_params.json", name)))
  write_series_csv(sigma(fit), file.path(OUTDIR, sprintf("fit_%s_sigma.csv", name)), "sigma")
  cat(sprintf("  fit %-14s loglik=%.4f\n", name, as.numeric(llhood(fit))))
}

garch_fit <- garch(returns, orders = c(1, 1), cond_dist = "norm", parallel = FALSE)
fit_and_dump(garch_fit, "garch11_norm", "garch", "norm")

egarch_fit <- fEGarch(egarch_spec(orders = c(1, 1), cond_dist = "norm"), returns, parallel = FALSE)
fit_and_dump(egarch_fit, "egarch11_norm", "egarch", "norm")

# --- Phase-1 short-memory models: GJR-GARCH, TGARCH, APARCH -------------------
# The v1.0.6 public signatures (confirmed via args(), NOT source) are data-first and
# identical to garch(): fn(rt, orders, cond_dist, ..., parallel) — so they are called
# exactly like garch above (aparch additionally has fix_delta = c(NA, 1, 2), i.e. delta
# is estimated by default, which we keep). Each fit is wrapped in tryCatch so a failure
# prints the model, the error and exists(<fn>) and NEVER writes a partial fixture.
tryCatch({
  gjr_fit <- gjrgarch(returns, orders = c(1, 1), cond_dist = "norm", parallel = FALSE)
  cat("  gjrgarch pars:", paste(names(pars(gjr_fit)), collapse = ", "), "\n")
  fit_and_dump(gjr_fit, "gjrgarch11_norm", "gjrgarch", "norm")
}, error = function(e) cat("  ERROR gjrgarch:", conditionMessage(e),
                          "| exists('gjrgarch') =", exists("gjrgarch"), "\n"))

tryCatch({
  tgarch_fit <- tgarch(returns, orders = c(1, 1), cond_dist = "norm", parallel = FALSE)
  cat("  tgarch pars:", paste(names(pars(tgarch_fit)), collapse = ", "), "\n")
  fit_and_dump(tgarch_fit, "tgarch11_norm", "tgarch", "norm")
}, error = function(e) cat("  ERROR tgarch:", conditionMessage(e),
                          "| exists('tgarch') =", exists("tgarch"), "\n"))

tryCatch({
  aparch_fit <- aparch(returns, orders = c(1, 1), cond_dist = "norm", parallel = FALSE)
  cat("  aparch pars:", paste(names(pars(aparch_fit)), collapse = ", "), "\n")
  fit_and_dump(aparch_fit, "aparch11_norm", "aparch", "norm")
}, error = function(e) cat("  ERROR aparch:", conditionMessage(e),
                          "| exists('aparch') =", exists("aparch"), "\n"))

# --- Phase-2 EGARCH-family (Type-II): Log-GARCH ------------------------------
# Log-GARCH is a Type-II EGF model, so it is spec-first (loggarch_spec() + fEGarch()),
# like egarch above — confirmed via ls()/args() (there is no data-first loggarch()
# function; no source read). Wrapped in tryCatch so a failure prints the model, the
# error and exists(loggarch_spec) and NEVER writes a partial fixture.
tryCatch({
  loggarch_fit <- fEGarch(loggarch_spec(orders = c(1, 1), cond_dist = "norm"), returns,
                          parallel = FALSE)
  cat("  loggarch pars:", paste(names(pars(loggarch_fit)), collapse = ", "), "\n")
  fit_and_dump(loggarch_fit, "loggarch11_norm", "loggarch", "norm")
}, error = function(e) cat("  ERROR loggarch:", conditionMessage(e),
                          "| exists('loggarch_spec') =", exists("loggarch_spec"), "\n"))

# --- Phase-2 EGARCH-family (Type-I): MEGARCH, MLog-GARCH ----------------------
# Both are Type-I EGF models with different modulus/power settings (WP171 Eqs. 7-9);
# they are spec-first (megarch_spec()/mloggarch_spec() + fEGarch()), like egarch —
# confirmed via ls()/args() (no data-first functions; no source read). Each wrapped in
# tryCatch so a failure prints the model, the error and exists(<spec>) and NEVER writes
# a partial fixture.
tryCatch({
  megarch_fit <- fEGarch(megarch_spec(orders = c(1, 1), cond_dist = "norm"), returns,
                         parallel = FALSE)
  cat("  megarch pars:", paste(names(pars(megarch_fit)), collapse = ", "), "\n")
  fit_and_dump(megarch_fit, "megarch11_norm", "megarch", "norm")
}, error = function(e) cat("  ERROR megarch:", conditionMessage(e),
                          "| exists('megarch_spec') =", exists("megarch_spec"), "\n"))

tryCatch({
  mloggarch_fit <- fEGarch(mloggarch_spec(orders = c(1, 1), cond_dist = "norm"), returns,
                           parallel = FALSE)
  cat("  mloggarch pars:", paste(names(pars(mloggarch_fit)), collapse = ", "), "\n")
  fit_and_dump(mloggarch_fit, "mloggarch11_norm", "mloggarch", "norm")
}, error = function(e) cat("  ERROR mloggarch:", conditionMessage(e),
                          "| exists('mloggarch_spec') =", exists("mloggarch_spec"), "\n"))

# --- Phase-4 long-memory: FIEGARCH -------------------------------------------
# FIEGARCH is the fractionally-integrated (long-memory) EGARCH: spec-first via the dedicated
# fiegarch_spec() wrapper + fEGarch() — confirmed via ls()/args() (there is NO long_memo flag on
# egarch_spec; the "fi" wrapper enables the fractional d; no source read). The fractional order d is
# an estimated parameter, and the fit uses the App. C.3 long-memory truncation default L = n-1.
# Wrapped in tryCatch so a failure prints the model, the error and exists(fiegarch_spec) and NEVER
# writes a partial fixture.
tryCatch({
  fiegarch_fit <- fEGarch(fiegarch_spec(orders = c(1, 1), cond_dist = "norm"), returns,
                          parallel = FALSE)
  cat("  fiegarch pars:", paste(names(pars(fiegarch_fit)), collapse = ", "), "\n")
  fit_and_dump(fiegarch_fit, "fiegarch11_norm", "fiegarch", "norm", trunc = "n-1")
}, error = function(e) cat("  ERROR fiegarch:", conditionMessage(e),
                          "| exists('fiegarch_spec') =", exists("fiegarch_spec"), "\n"))

# --- Phase-4 long-memory: FIMEGARCH, FIMLog-GARCH ----------------------------
# The fractionally-integrated Type-I modulus variants — long-memory analogues of MEGARCH /
# MLog-GARCH, same fractional order d as FIEGARCH. Spec-first via the dedicated fimegarch_spec() /
# fimloggarch_spec() wrappers + fEGarch(), like fiegarch_spec — the calling convention is CONFIRMED
# below via ls()/args() (NOT guessed; no source read). Diagnostic: list the package's fi*_spec
# wrappers and print the args of the two we use, so the console output pins the convention.
cat("  fi*_spec wrappers in package:fEGarch:",
    paste(grep("^fi.*_spec$", ls("package:fEGarch"), value = TRUE), collapse = ", "), "\n")
for (fn in c("fimegarch_spec", "fimloggarch_spec")) {
  if (exists(fn)) cat(sprintf("    args(%s): %s\n", fn,
                              paste(deparse(args(get(fn))), collapse = " ")))
}

# print the full named parameter vector (so the fractional order d is visible whatever it is named)
cat_pars <- function(fit) {
  p <- pars(fit)
  cat("   ", paste(sprintf("%s=%.7g", names(p), as.numeric(p)), collapse = ", "), "\n")
}

tryCatch({
  fimegarch_fit <- fEGarch(fimegarch_spec(orders = c(1, 1), cond_dist = "norm"), returns,
                           parallel = FALSE)
  cat("  fimegarch pars:", paste(names(pars(fimegarch_fit)), collapse = ", "), "\n")
  cat_pars(fimegarch_fit)
  fit_and_dump(fimegarch_fit, "fimegarch11_norm", "fimegarch", "norm", trunc = "n-1")
}, error = function(e) cat("  ERROR fimegarch:", conditionMessage(e),
                          "| exists('fimegarch_spec') =", exists("fimegarch_spec"), "\n"))

tryCatch({
  fimloggarch_fit <- fEGarch(fimloggarch_spec(orders = c(1, 1), cond_dist = "norm"), returns,
                             parallel = FALSE)
  cat("  fimloggarch pars:", paste(names(pars(fimloggarch_fit)), collapse = ", "), "\n")
  cat_pars(fimloggarch_fit)
  fit_and_dump(fimloggarch_fit, "fimloggarch11_norm", "fimloggarch", "norm", trunc = "n-1")
}, error = function(e) cat("  ERROR fimloggarch:", conditionMessage(e),
                          "| exists('fimloggarch_spec') =", exists("fimloggarch_spec"), "\n"))

# --- Phase-4 long-memory: FIGARCH (variance-recursion long memory) ------------
# FIGARCH is the fractionally-integrated GARCH — a VARIANCE-recursion long-memory model (the (1-L)^d
# operator enters the conditional-VARIANCE recursion, unlike the EGF theta(B) log-variance family).
# The fi*_spec listing above did NOT include figarch_spec, so FIGARCH is likely the data-first
# figarch() (like garch/gjrgarch/tgarch/aparch). The convention is CONFIRMED below via ls()/args()
# (NOT guessed; no source read): print the signature of whichever of figarch / figarch_spec exists,
# then call the one that does. Wrapped in tryCatch so a failure prints the model, the error and both
# exists() checks and NEVER writes a partial fixture.
cat("  figarch exists?", exists("figarch"), " figarch_spec exists?", exists("figarch_spec"), "\n")
for (fn in c("figarch", "figarch_spec")) {
  if (exists(fn)) cat(sprintf("    args(%s): %s\n", fn,
                              paste(deparse(args(get(fn))), collapse = " ")))
}

tryCatch({
  figarch_fit <- if (exists("figarch")) {
    figarch(returns, orders = c(1, 1), cond_dist = "norm", parallel = FALSE)
  } else {
    fEGarch(figarch_spec(orders = c(1, 1), cond_dist = "norm"), returns, parallel = FALSE)
  }
  cat("  figarch pars:", paste(names(pars(figarch_fit)), collapse = ", "), "\n")
  cat_pars(figarch_fit)
  # figarch()'s OWN defaults are trunc="none", presample=50 (from its args(), NOT the EGF FI models'
  # trunc="n-1") -- record FIGARCH's actual convention; the reconstruction gate resolves what "none"
  # means for the variance-recursion coefficient expansion.
  fit_and_dump(figarch_fit, "figarch11_norm", "figarch", "norm", trunc = "none")
}, error = function(e) cat("  ERROR figarch:", conditionMessage(e),
                          "| exists('figarch') =", exists("figarch"),
                          "| exists('figarch_spec') =", exists("figarch_spec"), "\n"))

# =============================================================================
# 3. Manifest — full provenance for every fixture.
# =============================================================================

manifest <- list(
  description = paste(
    "fEGarch OUTPUT fixtures for the quantica clean-room port (Phase 0).",
    "Generated by running fEGarch's PUBLIC functions and recording numeric output;",
    "no fEGarch source was read (CLAUDE.md §12). R is never a quantica/CI dependency."),
  fegarch_version = FEGARCH_VERSION,
  r_version = R_VERSION,
  generated_by = "tests/fixtures/fegarch/generate_fixtures.R",
  distribution_fixtures = list(
    method = paste(
      "fEGarch exposes standardized SAMPLERS (r*_s) but no public density/CDF/quantile",
      "functions, so distributions are characterised by sampler output: empirical moments",
      "and quantiles. Python compares its analytic values within Monte-Carlo tolerance."),
    n_samples = N_SAMPLES,
    base_seed = BASE_SEED,
    probs = PROBS,
    kurtosis = "raw (E[(x-mu)^4]/sd^4); standard normal ~ 3",
    files = c("distribution_moments.csv", "distribution_quantiles.csv"),
    labels = vapply(dist_specs, function(s) s$label, character(1)),
    reconcile = paste(
      "ALD uses fEGarch's `P` parameter (default 8) — the exact average-Laplace form is a",
      "RECONCILE item; the skew variants use fEGarch's `skew` (1 = symmetric). These fixtures",
      "are what lock the ALD form and the Fernandez-Steel normalization on the Python side.")),
  model_fixtures = list(
    input_series = list(
      file = "synthetic_returns.csv",
      note = "committed synthetic GARCH(1,1) series (NOT fEGarch's SP500 data — avoids redistributing its GPL-3 dataset)",
      seed = SIM_SEED, n = SIM_N, dgp = SIM, innovations = "standard normal"),
    settings = list(orders = c(1L, 1L), cond_dist = "norm", presample = 50L,
                    trunc = "none", mean_included = TRUE, parallel = FALSE),
    fits = list(
      garch11_norm = list(params = "fit_garch11_norm_params.json", sigma = "fit_garch11_norm_sigma.csv"),
      gjrgarch11_norm = list(params = "fit_gjrgarch11_norm_params.json", sigma = "fit_gjrgarch11_norm_sigma.csv"),
      tgarch11_norm = list(params = "fit_tgarch11_norm_params.json", sigma = "fit_tgarch11_norm_sigma.csv"),
      aparch11_norm = list(params = "fit_aparch11_norm_params.json", sigma = "fit_aparch11_norm_sigma.csv"),
      egarch11_norm = list(params = "fit_egarch11_norm_params.json", sigma = "fit_egarch11_norm_sigma.csv"),
      loggarch11_norm = list(params = "fit_loggarch11_norm_params.json", sigma = "fit_loggarch11_norm_sigma.csv"),
      megarch11_norm = list(params = "fit_megarch11_norm_params.json", sigma = "fit_megarch11_norm_sigma.csv"),
      mloggarch11_norm = list(params = "fit_mloggarch11_norm_params.json", sigma = "fit_mloggarch11_norm_sigma.csv"),
      fiegarch11_norm = list(params = "fit_fiegarch11_norm_params.json", sigma = "fit_fiegarch11_norm_sigma.csv",
                             note = "first Phase-4 long-memory fixture (fractionally-integrated EGARCH; fractional order d estimated, App. C.3 trunc L=n-1)"),
      fimegarch11_norm = list(params = "fit_fimegarch11_norm_params.json", sigma = "fit_fimegarch11_norm_sigma.csv",
                              note = "Phase-4 long-memory: fractionally-integrated MEGARCH (Type-I modulus asymmetry, |eta| magnitude; d estimated, App. C.3 trunc L=n-1)"),
      fimloggarch11_norm = list(params = "fit_fimloggarch11_norm_params.json", sigma = "fit_fimloggarch11_norm_sigma.csv",
                                note = "Phase-4 long-memory: fractionally-integrated MLog-GARCH (Type-I modulus asymmetry + ln(|eta|+1) magnitude; d estimated, App. C.3 trunc L=n-1)"),
      figarch11_norm = list(params = "fit_figarch11_norm_params.json", sigma = "fit_figarch11_norm_sigma.csv",
                            note = "Phase-4 long-memory: fractionally-integrated GARCH (VARIANCE-recursion long memory, NOT the EGF log-variance family; d estimated; figarch()'s own defaults trunc='none', presample=50)"))),
  pending_fixtures = paste(
    "Phase-2 EGARCH-family (1,1)/norm fits are complete; the Type-I long-memory FI models",
    "(FIEGARCH/FIMEGARCH/FIMLog-GARCH) are done, and FIGARCH (the variance-recursion long-memory GARCH)",
    "is added. Later phases need more fixtures: all short-memory + EGARCH-family models under the other",
    "7 conditional distributions; the remaining long-memory fits (FIAPARCH/FITGARCH/FIGJR/FILog-GARCH,",
    "Phase 4); dual-mean (ARMA/FARIMA) fits (Phase 5); and forecasts / VaR-ES (Phase 6). Extend this",
    "script and re-run when those models are implemented."))
write_json(manifest, file.path(OUTDIR, "manifest.json"))

cat("done.\n")
