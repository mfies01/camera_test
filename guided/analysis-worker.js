// Runs the live checks off the main thread. Receives a scaled-down ImageBitmap of the camera
// frame, returns the analysis result. Keeps the previous frame for the shake measurement.
"use strict";
importScripts("shelf-analysis.js?v=2026-11-19f");
let canvas = null, ctx = null, prevGray = null;
self.onmessage = e => {
  const { bmp, W, H } = e.data;
  if (!canvas || canvas.width !== W || canvas.height !== H) {
    canvas = new OffscreenCanvas(W, H);
    ctx = canvas.getContext("2d", { willReadFrequently: true });
    prevGray = null;
  }
  const t0 = performance.now();
  ctx.drawImage(bmp, 0, 0); bmp.close();
  const rgba = ctx.getImageData(0, 0, W, H).data;
  const t1 = performance.now();
  const le = ShelfAnalysis.lumaAndExposure(rgba, W * H);
  const r = ShelfAnalysis.analyzeFrame(le.gray, W, H, prevGray, le.exposure);
  prevGray = le.gray;
  r.workerTimes = { read: t1 - t0, analyze: performance.now() - t1 };
  self.postMessage(r);
};
