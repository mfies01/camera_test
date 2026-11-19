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
  // (x, y) has offset c = y*cos(phi) - x*sin(phi). Edge pixels vote with their gradient strength.
  function shelfLines(g, W, H) {
    const nA = Math.round(2 * MAX_LINE_DEG / STEP_DEG) + 1;
    const angles = [], cosA = [], sinA = [];
    for (let k = 0; k < nA; k++) {
      const a = -MAX_LINE_DEG + k * STEP_DEG; angles.push(a);
      cosA.push(Math.cos(a * Math.PI / 180)); sinA.push(Math.sin(a * Math.PI / 180));
    }
    // Sobel, keep pixels whose gradient points mostly vertically (= edge runs horizontally).
    const mags = [], pts = [];
    for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      const gx = (g[i - W + 1] + 2 * g[i + 1] + g[i + W + 1]) - (g[i - W - 1] + 2 * g[i - 1] + g[i + W - 1]);
      const gy = (g[i + W - 1] + 2 * g[i + W] + g[i + W + 1]) - (g[i - W - 1] + 2 * g[i - W] + g[i - W + 1]);
      const ax = Math.abs(gx), ay = Math.abs(gy);
      if (ay > 2.2 * ax && ay > 40) { mags.push(ay); pts.push(x, y); }
    }
    if (pts.length < W) return [];
    // Keep the strongest 40 % of candidate edge pixels: shelf edges are high-contrast.
    const sorted = Float32Array.from(mags).sort();
    const thr = sorted[Math.floor(sorted.length * 0.6)];
    const cMax = Math.ceil(Math.hypot(W, H)), cOff = cMax, nC = 2 * cMax + 1;
    const acc = new Float32Array(nA * nC);
    for (let p = 0, q = 0; p < mags.length; p++, q += 2) {
      if (mags[p] < thr) continue;
      const x = pts[q], y = pts[q + 1];
      for (let k = 0; k < nA; k++) {
        const c = Math.round(y * cosA[k] - x * sinA[k]) + cOff;
        acc[k * nC + c] += 1;
      }
    }
    // Peaks: one vote per pixel, so a peak's value is the number of edge pixels on the line.
    // A shelf edge has to cover at least 45 % of the width.
    const minVotes = 0.45 * W, lines = [];
    const used = new Uint8Array(nA * nC);
    for (;;) {
      let best = -1, bv = minVotes;
      for (let i = 0; i < acc.length; i++) if (!used[i] && acc[i] > bv) { bv = acc[i]; best = i; }
      if (best < 0 || lines.length >= 12) break;
      const k = Math.floor(best / nC), c = (best % nC) - cOff;
      // suppress neighbours: +-6 px offset, +-4 degrees
      for (let dk = -8; dk <= 8; dk++) for (let dc = -6; dc <= 6; dc++) {
        const kk = k + dk, cc = c + cOff + dc;
        if (kk >= 0 && kk < nA && cc >= 0 && cc < nC) used[kk * nC + cc] = 1;
      }
      const a = angles[k];
      // y of the line at the horizontal centre of the frame
      const yMid = (c + (W / 2) * sinA[k]) / cosA[k];
      if (yMid < 2 || yMid > H - 3) continue;
      lines.push({ angleDeg: a, yMid, votes: bv, coverage: bv / W });
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

  function analyzeFrame(gray, W, H, prevGray) {
    const ex = exposure(gray);
    const lines = shelfLines(gray, W, H);
    const rs = rollAndSkew(lines, H);
    return {
      sharpness: laplacianVariance(gray, W, H),
      clippedPct: ex.clippedPct, darkPct: ex.darkPct, meanLuma: ex.meanLuma,
      motion: motion(gray, prevGray),
      lines, rollDeg: rs.rollDeg, skewDeg: rs.skewDeg,
    };
  }

  const api = { analyzeFrame, lumaFromRGBA, laplacianVariance, shelfLines, rollAndSkew };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ShelfAnalysis = api;
})(typeof self !== "undefined" ? self : this);
