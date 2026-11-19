// Live checks for guided shelf capture. Pure functions on a small grayscale frame, no DOM,
// so the same code runs in the page and in the offline test (test/check_analysis.py).
//
//   analyzeFrame(gray, W, H, prevGray) -> { sharpness, clippedPct, darkPct, meanLuma,
//                                           motion, lines, rollDeg, skewDeg }
//
// gray: Float32Array of luma 0..255, row-major, W x H (the page uses W = 320).
// Shelf edges are long, near-horizontal lines. Their mean angle is the roll of the picture;
// how their angle changes from the top to the bottom of the frame shows that the user stands
// off to one side (horizontal lines converge toward the far side of the shelf).
(function (root) {
  "use strict";

  const MAX_LINE_DEG = 20;     // search range for shelf-edge angles
  const STEP_DEG = 0.5;

  function lumaFromRGBA(rgba, n) {
    const g = new Float32Array(n);
    for (let i = 0, j = 0; j < n; i += 4, j++) g[j] = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
    return g;
  }

  // Grey frame and exposure statistics in one pass over the pixels (saves a loop on slow phones).
  function lumaAndExposure(rgba, n) {
    const g = new Float32Array(n);
    let clipped = 0, dark = 0, s = 0;
    for (let i = 0, j = 0; j < n; i += 4, j++) {
      const y = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
      g[j] = y; s += y;
      if (y >= 250) clipped++; else if (y <= 15) dark++;
    }
    return { gray: g, exposure: { clippedPct: 100 * clipped / n, darkPct: 100 * dark / n, meanLuma: s / n } };
  }

  function laplacianVariance(g, W, H) {
    let sum = 0, sum2 = 0, n = 0;
    for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
      const i = y * W + x, v = g[i - W] + g[i + W] + g[i - 1] + g[i + 1] - 4 * g[i];
      sum += v; sum2 += v * v; n++;
    }
    const m = sum / n; return sum2 / n - m * m;
  }

  function exposure(g) {
    let clipped = 0, dark = 0, s = 0;
    for (let i = 0; i < g.length; i++) { const y = g[i]; s += y; if (y >= 250) clipped++; else if (y <= 15) dark++; }
    return { clippedPct: 100 * clipped / g.length, darkPct: 100 * dark / g.length, meanLuma: s / g.length };
  }

  // Mean absolute difference to the previous frame, on a 4x subsampled grid.
  function motion(g, prev) {
    if (!prev || prev.length !== g.length) return null;
    let s = 0, n = 0;
    for (let i = 0; i < g.length; i += 4) { s += Math.abs(g[i] - prev[i]); n++; }
    return s / n;
  }

  // Hough transform restricted to near-horizontal lines.
  // A line at angle phi (degrees, positive = falling to the right on screen, y down) through
  // (x, y) has offset c = y*cos(phi) - x*sin(phi). Each edge pixel votes only for angles within
  // VOTE_WINDOW_DEG of its own edge direction (from the Sobel gradient), which keeps it cheap.
  const VOTE_WINDOW_DEG = 20;     // = whole search range; narrower windows lost shelf edges in the test
  let buf = null;                       // reused between frames
  function buffers(n, nAcc) {
    if (!buf || buf.n < n || buf.nAcc !== nAcc) {
      buf = { n, nAcc, mag: new Float32Array(n), x: new Int16Array(n), y: new Int16Array(n),
              k: new Int16Array(n), acc: new Float32Array(nAcc), hist: new Uint32Array(1024) };
    }
    return buf;
  }
  function shelfLines(g, W, H) {
    const nA = Math.round(2 * MAX_LINE_DEG / STEP_DEG) + 1;
    const angles = new Float32Array(nA), cosA = new Float32Array(nA), sinA = new Float32Array(nA);
    for (let k = 0; k < nA; k++) {
      const a = -MAX_LINE_DEG + k * STEP_DEG; angles[k] = a;
      cosA[k] = Math.cos(a * Math.PI / 180); sinA[k] = Math.sin(a * Math.PI / 180);
    }
    const cMax = Math.ceil(Math.hypot(W, H)), cOff = cMax, nC = 2 * cMax + 1;
    const b = buffers(W * H, nA * nC), mag = b.mag, X = b.x, Y = b.y, K = b.k, hist = b.hist;
    hist.fill(0);
    // Sobel, keep pixels whose gradient points mostly vertically (= edge runs horizontally).
    let n = 0;
    for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      const gx = (g[i - W + 1] + 2 * g[i + 1] + g[i + W + 1]) - (g[i - W - 1] + 2 * g[i - 1] + g[i + W - 1]);
      const gy = (g[i + W - 1] + 2 * g[i + W] + g[i + W + 1]) - (g[i - W - 1] + 2 * g[i - W] + g[i - W + 1]);
      const ax = gx < 0 ? -gx : gx, ay = gy < 0 ? -gy : gy;
      if (ay > 2.2 * ax && ay > 40) {
        mag[n] = ay; X[n] = x; Y[n] = y;
        // edge direction: tangent (1, tan phi) is perpendicular to (gx, gy) -> tan phi = -gx / gy
        K[n] = Math.round((Math.atan(-gx / gy) * 57.29578 + MAX_LINE_DEG) / STEP_DEG);
        hist[ay > 1023 ? 1023 : ay | 0]++;
        n++;
      }
    }
    if (n < W) return [];
    // Keep the strongest 40 % of candidate edge pixels: shelf edges are high-contrast.
    let thr = 1023;
    for (let cum = 0; thr > 0; thr--) { cum += hist[thr]; if (cum >= 0.4 * n) break; }
    const acc = b.acc; acc.fill(0);
    const win = Math.round(VOTE_WINDOW_DEG / STEP_DEG);
    for (let p = 0; p < n; p++) {
      if (mag[p] < thr) continue;
      const x = X[p], y = Y[p], k0 = K[p];
      const lo = k0 - win < 0 ? 0 : k0 - win, hi = k0 + win > nA - 1 ? nA - 1 : k0 + win;
      for (let k = lo; k <= hi; k++) acc[k * nC + Math.round(y * cosA[k] - x * sinA[k]) + cOff] += 1;
    }
    // Peaks: one vote per pixel, so a peak's value is the number of edge pixels on the line.
    // A shelf edge has to cover at least 45 % of the width. Greedy pick, strongest first,
    // skipping candidates within +-6 px offset and +-4 degrees of an accepted line.
    const minVotes = 0.45 * W, cand = [];
    for (let i = 0; i < acc.length; i++) if (acc[i] > minVotes) cand.push(i);
    cand.sort((a, c) => acc[c] - acc[a]);
    const picked = [], lines = [];
    for (const i of cand) {
      if (picked.length >= 12) break;
      const k = Math.floor(i / nC), c = (i % nC) - cOff;
      if (picked.some(q => Math.abs(q.k - k) <= 8 && Math.abs(q.c - c) <= 6)) continue;
      picked.push({ k, c });
      const a = angles[k];
      // y of the line at the horizontal centre of the frame
      const yMid = (c + (W / 2) * sinA[k]) / cosA[k];
      if (yMid < 2 || yMid > H - 3) continue;
      lines.push({ angleDeg: a, yMid, votes: acc[i], coverage: acc[i] / W });
    }
    return lines.sort((p, q) => p.yMid - q.yMid);
  }

  // roll: vote-weighted median angle of the shelf edges.
  // skew: angle change from top to bottom of the frame (least squares of angle over yMid / H).
  // Facing the shelf squarely gives skew ~ 0. Positive skew = lines open toward the right =
  // the right side of the shelf is nearer; negative = the right side is farther away.
  function rollAndSkew(lines, H) {
    if (!lines.length) return { rollDeg: null, skewDeg: null };
    const byAngle = lines.slice().sort((a, b) => a.angleDeg - b.angleDeg);
    const total = byAngle.reduce((s, l) => s + l.votes, 0);
    let acc = 0, roll = byAngle[0].angleDeg;
    for (const l of byAngle) { acc += l.votes; if (acc >= total / 2) { roll = l.angleDeg; break; } }
    if (lines.length < 3) return { rollDeg: roll, skewDeg: null };
    let sw = 0, sy = 0, sa = 0;
    for (const l of lines) { const w = l.votes; sw += w; sy += w * l.yMid / H; sa += w * l.angleDeg; }
    const my = sy / sw, ma = sa / sw;
    let num = 0, den = 0;
    for (const l of lines) { const w = l.votes, dy = l.yMid / H - my; num += w * dy * (l.angleDeg - ma); den += w * dy * dy; }
    if (den < 0.02 * sw) return { rollDeg: roll, skewDeg: null };  // lines too close together to tell
    return { rollDeg: roll, skewDeg: num / den };
  }

  function halfSize(g, W, H) {
    const w = W >> 1, h = H >> 1, o = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = 2 * y * W + 2 * x;
      o[y * w + x] = 0.25 * (g[i] + g[i + 1] + g[i + W] + g[i + W + 1]);
    }
    return { g: o, W: w, H: h };
  }

  // ex: exposure from lumaAndExposure, if already computed
  function analyzeFrame(gray, W, H, prevGray, ex) {
    const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
    const t0 = now();
    const sharpness = laplacianVariance(gray, W, H);
    const t1 = now();
    if (!ex) ex = exposure(gray);
    const mo = motion(gray, prevGray);
    const t2 = now();
    // Shelf edges are long, coarse structures: find them on a half-size frame (a quarter of the
    // pixels), then scale back to the analysis frame.
    const half = halfSize(gray, W, H);
    const lines = shelfLines(half.g, half.W, half.H).map(l =>
      ({ angleDeg: l.angleDeg, yMid: l.yMid * H / half.H, votes: l.votes, coverage: l.coverage }));
    const rs = rollAndSkew(lines, H);
    const t3 = now();
    return {
      sharpness,
      clippedPct: ex.clippedPct, darkPct: ex.darkPct, meanLuma: ex.meanLuma,
      motion: mo,
      lines, rollDeg: rs.rollDeg, skewDeg: rs.skewDeg,
      parts: { sharp: t1 - t0, exposure: t2 - t1, lines: t3 - t2 },
    };
  }

  const api = { VERSION: "2026-11-19f", analyzeFrame, lumaFromRGBA, lumaAndExposure, laplacianVariance, shelfLines, rollAndSkew };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ShelfAnalysis = api;
})(typeof self !== "undefined" ? self : this);
