// Pixel-based perception: HSV color segmentation on the overhead camera image
// -> object centroids in robot coordinates. Works on the simulated camera
// (RobotScene.captureVision) and on any RGBA frame with a known top-down
// pixel->world mapping (e.g. a calibrated phone/USB camera over a real table).

export const HUE_RANGES = {
  red: [[345, 360], [0, 12]], yellow: [[40, 70]], green: [[95, 150]],
  blue: [[195, 235]], purple: [[270, 310]],
};

export function rgb2hsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  let h = 0;
  if (d > 0) {
    if (mx === r) h = 60 * (((g - b) / d) % 6);
    else if (mx === g) h = 60 * ((b - r) / d + 2);
    else h = 60 * ((r - g) / d + 4);
  }
  if (h < 0) h += 360;
  return [h, mx === 0 ? 0 : d / mx, mx];
}

const inHue = (h, ranges) => ranges.some(([a, b]) => h >= a && h <= b);

/**
 * @param {Uint8Array} px RGBA, width*height*4, row 0 = bottom (WebGL order)
 * @param {{xMin,xMax,yMin,yMax}} bounds world rectangle covered by the image
 * @returns {Record<string, {x:number,y:number,pixels:number,bbox:number[]}>}
 */
export function detect(px, width, height, bounds, { minPixels = 12, sMin = 0.45, vMin = 0.25 } = {}) {
  const acc = {};
  for (const c of Object.keys(HUE_RANGES)) acc[c] = { n: 0, su: 0, sv: 0, u0: 1e9, v0: 1e9, u1: -1, v1: -1 };
  for (let v = 0; v < height; v++) {
    for (let u = 0; u < width; u++) {
      const i = (v * width + u) * 4;
      const [h, s, val] = rgb2hsv(px[i], px[i + 1], px[i + 2]);
      if (s < sMin || val < vMin) continue;
      for (const [c, ranges] of Object.entries(HUE_RANGES)) {
        if (!inHue(h, ranges)) continue;
        const a = acc[c];
        a.n++; a.su += u; a.sv += v;
        a.u0 = Math.min(a.u0, u); a.v0 = Math.min(a.v0, v); a.u1 = Math.max(a.u1, u); a.v1 = Math.max(a.v1, v);
      }
    }
  }
  const out = {};
  for (const [c, a] of Object.entries(acc)) {
    if (a.n < minPixels) continue;
    const u = a.su / a.n, v = a.sv / a.n;
    // orthographic camera, image up = +x, image right = -y (see RobotScene.visionCam)
    const x = bounds.xMin + ((v + 0.5) / height) * (bounds.xMax - bounds.xMin);
    const y = bounds.yMax - ((u + 0.5) / width) * (bounds.yMax - bounds.yMin);
    out[c] = { x, y, pixels: a.n, bbox: [a.u0, a.v0, a.u1, a.v1] };
  }
  return out;
}

/** Draw the camera frame + detections into a 2D canvas (flips WebGL row order). */
export function drawDetections(canvas, px, width, height, dets) {
  const ctx = canvas.getContext("2d");
  canvas.width = width; canvas.height = height;
  const img = ctx.createImageData(width, height);
  for (let v = 0; v < height; v++) {
    const src = (height - 1 - v) * width * 4;
    img.data.set(px.subarray(src, src + width * 4), v * width * 4);
  }
  ctx.putImageData(img, 0, 0);
  ctx.lineWidth = 1;
  ctx.font = "9px sans-serif";
  for (const [c, d] of Object.entries(dets)) {
    const [u0, v0, u1, v1] = d.bbox;
    ctx.strokeStyle = "#fff";
    ctx.strokeRect(u0 - 1, height - 1 - v1 - 1, u1 - u0 + 2, v1 - v0 + 2);
    ctx.fillStyle = "#fff";
    ctx.fillText(c, u0, Math.max(8, height - 1 - v1 - 3));
  }
}
