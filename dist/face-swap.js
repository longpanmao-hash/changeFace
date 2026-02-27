"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const faceapi = __importStar(require("@vladmandic/face-api/dist/face-api.node-wasm"));
const canvas_1 = require("canvas");
const commander_1 = require("commander");
const delaunator_1 = __importDefault(require("delaunator"));
const promises_1 = __importDefault(require("node:fs/promises"));
const node_path_1 = __importDefault(require("node:path"));
faceapi.env.monkeyPatch({ Canvas: canvas_1.Canvas, Image: canvas_1.Image, ImageData: canvas_1.ImageData });
async function ensureFile(filePath, label) {
    try {
        await promises_1.default.access(filePath);
    }
    catch {
        throw new Error(`${label} not found: ${filePath}`);
    }
}
function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
}
function smoothstep(edge0, edge1, x) {
    const t = clamp((x - edge0) / Math.max(1e-6, edge1 - edge0), 0, 1);
    return t * t * (3 - 2 * t);
}
function solveLinearSystem6x6(A, b) {
    const n = 6;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let i = 0; i < n; i += 1) {
        let pivot = i;
        for (let r = i + 1; r < n; r += 1) {
            if (Math.abs(M[r][i]) > Math.abs(M[pivot][i]))
                pivot = r;
        }
        if (Math.abs(M[pivot][i]) < 1e-10)
            throw new Error("Singular system while fitting affine matrix");
        if (pivot !== i) {
            const t = M[i];
            M[i] = M[pivot];
            M[pivot] = t;
        }
        const div = M[i][i];
        for (let c = i; c <= n; c += 1)
            M[i][c] /= div;
        for (let r = 0; r < n; r += 1) {
            if (r === i)
                continue;
            const f = M[r][i];
            for (let c = i; c <= n; c += 1)
                M[r][c] -= f * M[i][c];
        }
    }
    return M.map((row) => row[n]);
}
function computeAffineLeastSquares(src, dst) {
    if (src.length !== dst.length || src.length < 3)
        throw new Error("Need at least 3 matching points");
    const ATA = Array.from({ length: 6 }, () => Array(6).fill(0));
    const ATb = Array(6).fill(0);
    const addRow = (row, v) => {
        for (let i = 0; i < 6; i += 1) {
            ATb[i] += row[i] * v;
            for (let j = 0; j < 6; j += 1)
                ATA[i][j] += row[i] * row[j];
        }
    };
    for (let i = 0; i < src.length; i += 1) {
        const s = src[i];
        const d = dst[i];
        addRow([s.x, s.y, 1, 0, 0, 0], d.x);
        addRow([0, 0, 0, s.x, s.y, 1], d.y);
    }
    const [a, c, e, b, d, f] = solveLinearSystem6x6(ATA, ATb);
    return { a, b, c, d, e, f };
}
function averagePoint(points) {
    let x = 0;
    let y = 0;
    for (const p of points) {
        x += p.x;
        y += p.y;
    }
    return { x: x / points.length, y: y / points.length };
}
function getLandmarkPoints(landmarks) {
    return landmarks.positions.map((p) => ({ x: p.x, y: p.y }));
}
function getForeheadPoints(landmarks) {
    const leftBrow = landmarks.getLeftEyeBrow().map((p) => ({ x: p.x, y: p.y }));
    const rightBrow = landmarks.getRightEyeBrow().map((p) => ({ x: p.x, y: p.y }));
    const leftEye = averagePoint(landmarks.getLeftEye().map((p) => ({ x: p.x, y: p.y })));
    const rightEye = averagePoint(landmarks.getRightEye().map((p) => ({ x: p.x, y: p.y })));
    const browArc = [...rightBrow.slice().reverse(), ...leftBrow.slice().reverse()];
    return browArc.map((p, i) => {
        const t = i / Math.max(1, browArc.length - 1);
        const mid = 1 - Math.abs(t - 0.5) * 2;
        const eye = p.x < (leftEye.x + rightEye.x) / 2 ? leftEye : rightEye;
        const k = 0.8 * (0.35 + 0.65 * mid);
        return {
            x: p.x + (p.x - eye.x) * k,
            y: p.y + (p.y - eye.y) * k
        };
    });
}
function getFacePolygon(landmarks) {
    const jaw = landmarks.getJawOutline().map((p) => ({ x: p.x, y: p.y }));
    const forehead = getForeheadPoints(landmarks);
    return [...jaw, ...forehead];
}
function getFeaturePolygons(landmarks) {
    const leftEye = landmarks.getLeftEye().map((p) => ({ x: p.x, y: p.y }));
    const rightEye = landmarks.getRightEye().map((p) => ({ x: p.x, y: p.y }));
    const mouth = landmarks.getMouth().map((p) => ({ x: p.x, y: p.y }));
    const nose = landmarks
        .getNose()
        .slice(2, 9)
        .map((p) => ({ x: p.x, y: p.y }));
    return [leftEye, rightEye, mouth, nose];
}
function getAugmentedMesh(landmarks) {
    return [...getLandmarkPoints(landmarks), ...getForeheadPoints(landmarks)];
}
function computeBounds(points, width, height, padding = 20) {
    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (const p of points) {
        if (p.x < minX)
            minX = p.x;
        if (p.y < minY)
            minY = p.y;
        if (p.x > maxX)
            maxX = p.x;
        if (p.y > maxY)
            maxY = p.y;
    }
    const x0 = clamp(Math.floor(minX - padding), 0, width - 1);
    const y0 = clamp(Math.floor(minY - padding), 0, height - 1);
    const x1 = clamp(Math.ceil(maxX + padding), 1, width);
    const y1 = clamp(Math.ceil(maxY + padding), 1, height);
    return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
}
function pointInPolygon(p, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const pi = poly[i];
        const pj = poly[j];
        const intersect = pi.y > p.y !== pj.y > p.y && p.x < ((pj.x - pi.x) * (p.y - pi.y)) / (pj.y - pi.y + 1e-8) + pi.x;
        if (intersect)
            inside = !inside;
    }
    return inside;
}
function distancePointToSegment(p, a, b) {
    const abx = b.x - a.x;
    const aby = b.y - a.y;
    const apx = p.x - a.x;
    const apy = p.y - a.y;
    const denom = abx * abx + aby * aby + 1e-8;
    const t = clamp((apx * abx + apy * aby) / denom, 0, 1);
    const x = a.x + t * abx;
    const y = a.y + t * aby;
    const dx = p.x - x;
    const dy = p.y - y;
    return Math.sqrt(dx * dx + dy * dy);
}
function buildSoftMask(width, height, polygon, feather = 16) {
    const mask = (0, canvas_1.createCanvas)(width, height);
    const ctx = mask.getContext("2d");
    ctx.fillStyle = "rgba(0,0,0,0)";
    ctx.fillRect(0, 0, width, height);
    const bounds = computeBounds(polygon, width, height, feather + 2);
    const img = ctx.getImageData(bounds.x, bounds.y, bounds.width, bounds.height);
    const d = img.data;
    for (let y = 0; y < bounds.height; y += 1) {
        for (let x = 0; x < bounds.width; x += 1) {
            const px = x + bounds.x;
            const py = y + bounds.y;
            const idx = (y * bounds.width + x) * 4;
            const inside = pointInPolygon({ x: px + 0.5, y: py + 0.5 }, polygon);
            if (!inside)
                continue;
            let edgeDist = Number.POSITIVE_INFINITY;
            for (let i = 0; i < polygon.length; i += 1) {
                const a = polygon[i];
                const b = polygon[(i + 1) % polygon.length];
                const dist = distancePointToSegment({ x: px + 0.5, y: py + 0.5 }, a, b);
                if (dist < edgeDist)
                    edgeDist = dist;
            }
            const alpha = clamp(edgeDist / feather, 0, 1);
            d[idx] = 255;
            d[idx + 1] = 255;
            d[idx + 2] = 255;
            d[idx + 3] = Math.round(alpha * 255);
        }
    }
    ctx.putImageData(img, bounds.x, bounds.y);
    return mask;
}
function buildFeatureMask(width, height, polygons) {
    const mask = (0, canvas_1.createCanvas)(width, height);
    const ctx = mask.getContext("2d");
    ctx.fillStyle = "rgba(0,0,0,0)";
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = "rgba(255,255,255,1)";
    ctx.beginPath();
    ctx.moveTo(polygons[0].x, polygons[0].y);
    for (let i = 1; i < polygons.length; i += 1)
        ctx.lineTo(polygons[i].x, polygons[i].y);
    ctx.closePath();
    ctx.fill();
    return mask;
}
function drawWarpedByTriangles(sourceCanvas, srcPts, dstPts, facePoly, width, height) {
    const warped = (0, canvas_1.createCanvas)(width, height);
    const wctx = warped.getContext("2d");
    const delaunay = delaunator_1.default.from(dstPts.map((p) => [p.x, p.y]));
    const t = delaunay.triangles;
    for (let i = 0; i < t.length; i += 3) {
        const i0 = t[i];
        const i1 = t[i + 1];
        const i2 = t[i + 2];
        const d0 = dstPts[i0];
        const d1 = dstPts[i1];
        const d2 = dstPts[i2];
        const centroid = { x: (d0.x + d1.x + d2.x) / 3, y: (d0.y + d1.y + d2.y) / 3 };
        if (!pointInPolygon(centroid, facePoly))
            continue;
        const s0 = srcPts[i0];
        const s1 = srcPts[i1];
        const s2 = srcPts[i2];
        const m = computeAffineLeastSquares([s0, s1, s2], [d0, d1, d2]);
        wctx.save();
        wctx.beginPath();
        wctx.moveTo(d0.x, d0.y);
        wctx.lineTo(d1.x, d1.y);
        wctx.lineTo(d2.x, d2.y);
        wctx.closePath();
        wctx.clip();
        wctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
        wctx.drawImage(sourceCanvas, 0, 0);
        wctx.restore();
    }
    return warped;
}
function rgbToLab(r, g, b) {
    const srgb = [r, g, b].map((v) => {
        const x = v / 255;
        return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
    });
    const x = srgb[0] * 0.4124564 + srgb[1] * 0.3575761 + srgb[2] * 0.1804375;
    const y = srgb[0] * 0.2126729 + srgb[1] * 0.7151522 + srgb[2] * 0.072175;
    const z = srgb[0] * 0.0193339 + srgb[1] * 0.119192 + srgb[2] * 0.9503041;
    const xn = 0.95047;
    const yn = 1;
    const zn = 1.08883;
    const f = (t) => (t > 0.008856 ? t ** (1 / 3) : 7.787 * t + 16 / 116);
    const fx = f(x / xn);
    const fy = f(y / yn);
    const fz = f(z / zn);
    return { l: 116 * fy - 16, a: 500 * (fx - fy), b: 200 * (fy - fz) };
}
function labToRgb(lab) {
    const fy = (lab.l + 16) / 116;
    const fx = lab.a / 500 + fy;
    const fz = fy - lab.b / 200;
    const invf = (t) => {
        const t3 = t * t * t;
        return t3 > 0.008856 ? t3 : (t - 16 / 116) / 7.787;
    };
    const xn = 0.95047;
    const yn = 1;
    const zn = 1.08883;
    const x = xn * invf(fx);
    const y = yn * invf(fy);
    const z = zn * invf(fz);
    let r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
    let g = x * -0.969266 + y * 1.8760108 + z * 0.041556;
    let b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;
    const gamma = (v) => (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055);
    r = clamp(gamma(r), 0, 1);
    g = clamp(gamma(g), 0, 1);
    b = clamp(gamma(b), 0, 1);
    return { r: Math.round(r * 255), g: Math.round(g * 255), b: Math.round(b * 255) };
}
function isLikelySkin(r, g, b) {
    const maxc = Math.max(r, g, b);
    const minc = Math.min(r, g, b);
    if (r < 40 || g < 20 || b < 15)
        return false;
    if (maxc - minc < 15)
        return false;
    if (r < g || r < b)
        return false;
    const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
    const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
    return cr >= 133 && cr <= 173 && cb >= 77 && cb <= 127;
}
function blendFace(sourceCanvas, sourceDetection, targetCanvas, targetDetection, strength) {
    const width = targetCanvas.width;
    const height = targetCanvas.height;
    const result = (0, canvas_1.createCanvas)(width, height);
    const rctx = result.getContext("2d");
    rctx.drawImage(targetCanvas, 0, 0);
    const srcPts = getAugmentedMesh(sourceDetection.landmarks);
    const dstPts = getAugmentedMesh(targetDetection.landmarks);
    const facePoly = getFacePolygon(targetDetection.landmarks);
    const featurePolys = getFeaturePolygons(targetDetection.landmarks);
    const warped = drawWarpedByTriangles(sourceCanvas, srcPts, dstPts, facePoly, width, height);
    const mask = buildSoftMask(width, height, facePoly, 30);
    const featureMaskCanvas = (0, canvas_1.createCanvas)(width, height);
    const featureCtx = featureMaskCanvas.getContext("2d");
    for (const poly of featurePolys) {
        const fm = buildFeatureMask(width, height, poly);
        featureCtx.globalAlpha = 1;
        featureCtx.drawImage(fm, 0, 0);
    }
    const bounds = computeBounds(facePoly, width, height, 36);
    const targetData = rctx.getImageData(bounds.x, bounds.y, bounds.width, bounds.height);
    const warpedData = warped.getContext("2d").getImageData(bounds.x, bounds.y, bounds.width, bounds.height);
    const maskData = mask.getContext("2d").getImageData(bounds.x, bounds.y, bounds.width, bounds.height);
    const featureData = featureMaskCanvas.getContext("2d").getImageData(bounds.x, bounds.y, bounds.width, bounds.height);
    const td = targetData.data;
    const wd = warpedData.data;
    const md = maskData.data;
    const fd = featureData.data;
    let srcCount = 0;
    let tarCount = 0;
    let srcMean = { l: 0, a: 0, b: 0 };
    let tarMean = { l: 0, a: 0, b: 0 };
    for (let i = 0; i < wd.length; i += 4) {
        const m = md[i + 3] / 255;
        if (m < 0.15)
            continue;
        const ws = isLikelySkin(wd[i], wd[i + 1], wd[i + 2]);
        const ts = isLikelySkin(td[i], td[i + 1], td[i + 2]);
        if (wd[i + 3] > 5 && ws) {
            const sl = rgbToLab(wd[i], wd[i + 1], wd[i + 2]);
            srcMean.l += sl.l;
            srcMean.a += sl.a;
            srcMean.b += sl.b;
            srcCount += 1;
        }
        if (ts) {
            const tl = rgbToLab(td[i], td[i + 1], td[i + 2]);
            tarMean.l += tl.l;
            tarMean.a += tl.a;
            tarMean.b += tl.b;
            tarCount += 1;
        }
    }
    if (srcCount === 0 || tarCount === 0)
        return result;
    srcMean = { l: srcMean.l / srcCount, a: srcMean.a / srcCount, b: srcMean.b / srcCount };
    tarMean = { l: tarMean.l / tarCount, a: tarMean.a / tarCount, b: tarMean.b / tarCount };
    let srcVar = { l: 0, a: 0, b: 0 };
    let tarVar = { l: 0, a: 0, b: 0 };
    for (let i = 0; i < wd.length; i += 4) {
        const m = md[i + 3] / 255;
        if (m < 0.15)
            continue;
        const ws = isLikelySkin(wd[i], wd[i + 1], wd[i + 2]);
        const ts = isLikelySkin(td[i], td[i + 1], td[i + 2]);
        if (wd[i + 3] > 5 && ws) {
            const sl = rgbToLab(wd[i], wd[i + 1], wd[i + 2]);
            srcVar.l += (sl.l - srcMean.l) ** 2;
            srcVar.a += (sl.a - srcMean.a) ** 2;
            srcVar.b += (sl.b - srcMean.b) ** 2;
        }
        if (ts) {
            const tl = rgbToLab(td[i], td[i + 1], td[i + 2]);
            tarVar.l += (tl.l - tarMean.l) ** 2;
            tarVar.a += (tl.a - tarMean.a) ** 2;
            tarVar.b += (tl.b - tarMean.b) ** 2;
        }
    }
    const srcStd = {
        l: Math.sqrt(srcVar.l / Math.max(1, srcCount)),
        a: Math.sqrt(srcVar.a / Math.max(1, srcCount)),
        b: Math.sqrt(srcVar.b / Math.max(1, srcCount))
    };
    const tarStd = {
        l: Math.sqrt(tarVar.l / Math.max(1, tarCount)),
        a: Math.sqrt(tarVar.a / Math.max(1, tarCount)),
        b: Math.sqrt(tarVar.b / Math.max(1, tarCount))
    };
    const scale = {
        l: clamp(tarStd.l / Math.max(1e-4, srcStd.l), 0.6, 1.6),
        a: clamp(tarStd.a / Math.max(1e-4, srcStd.a), 0.7, 1.5),
        b: clamp(tarStd.b / Math.max(1e-4, srcStd.b), 0.7, 1.5)
    };
    for (let i = 0; i < wd.length; i += 4) {
        const m = md[i + 3] / 255;
        if (m < 0.01 || wd[i + 3] < 5)
            continue;
        const ws = isLikelySkin(wd[i], wd[i + 1], wd[i + 2]);
        const localPixel = i / 4;
        const py = Math.floor(localPixel / bounds.width);
        const yNorm = py / Math.max(1, bounds.height - 1);
        const foreheadFade = smoothstep(0.12, 0.35, yNorm);
        const fAlpha = fd[i + 3] / 255;
        const baseAlpha = m * (ws ? 0.9 : 0.22) * foreheadFade * strength;
        const featureAlpha = fAlpha * (0.35 + 0.6 * strength);
        const alpha = clamp(Math.max(baseAlpha, featureAlpha), 0, 1);
        const sl = rgbToLab(wd[i], wd[i + 1], wd[i + 2]);
        const mapped = {
            l: (sl.l - srcMean.l) * scale.l + tarMean.l,
            a: (sl.a - srcMean.a) * scale.a + tarMean.a,
            b: (sl.b - srcMean.b) * scale.b + tarMean.b
        };
        const targetLab = rgbToLab(td[i], td[i + 1], td[i + 2]);
        const preserveLight = 0.92 - 0.22 * strength;
        const merged = {
            l: targetLab.l * preserveLight + mapped.l * (1 - preserveLight),
            a: mapped.a,
            b: mapped.b
        };
        const rgb = fAlpha > 0.1 ? { r: wd[i], g: wd[i + 1], b: wd[i + 2] } : labToRgb(merged);
        td[i] = Math.round(rgb.r * alpha + td[i] * (1 - alpha));
        td[i + 1] = Math.round(rgb.g * alpha + td[i + 1] * (1 - alpha));
        td[i + 2] = Math.round(rgb.b * alpha + td[i + 2] * (1 - alpha));
        td[i + 3] = 255;
    }
    rctx.putImageData(targetData, bounds.x, bounds.y);
    return result;
}
async function loadModels(modelDir) {
    const tf = faceapi.tf;
    await tf.setBackend("wasm");
    await tf.ready();
    await faceapi.nets.tinyFaceDetector.loadFromDisk(modelDir);
    await faceapi.nets.faceLandmark68TinyNet.loadFromDisk(modelDir);
}
async function detectFace(imagePath) {
    const image = await (0, canvas_1.loadImage)(imagePath);
    const canvas = (0, canvas_1.createCanvas)(image.width, image.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0);
    const detection = await faceapi
        .detectSingleFace(canvas, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks(true);
    if (!detection)
        throw new Error(`No face detected: ${imagePath}`);
    return { canvas, detection };
}
async function main() {
    const program = new commander_1.Command();
    program
        .argument("[source]", "source image")
        .argument("[target]", "target image")
        .argument("[output]", "output image")
        .option("-s, --source <path>", "source image")
        .option("-t, --target <path>", "target image")
        .option("-o, --output <path>", "output image")
        .option("--strength <number>", "swap strength (0~1, lower keeps more target structure)")
        .option("-m, --models <path>", "models directory", node_path_1.default.resolve(process.cwd(), "models"));
    program.parse(process.argv);
    const options = program.opts();
    const [argSource, argTarget, argOutput, argStrength] = program.args;
    const sourceInput = options.source ?? argSource;
    const targetInput = options.target ?? argTarget;
    const outputInput = options.output ?? argOutput;
    if (!sourceInput || !targetInput || !outputInput)
        throw new Error("Missing source/target/output");
    const sourcePath = node_path_1.default.resolve(sourceInput);
    const targetPath = node_path_1.default.resolve(targetInput);
    const outputPath = node_path_1.default.resolve(outputInput);
    const modelDir = node_path_1.default.resolve(options.models);
    const strengthRaw = options.strength !== undefined ? options.strength : argStrength ?? "0.75";
    const strength = clamp(Number.parseFloat(strengthRaw), 0, 1);
    if (Number.isNaN(strength))
        throw new Error("Invalid --strength value");
    await ensureFile(sourcePath, "source image");
    await ensureFile(targetPath, "target image");
    await loadModels(modelDir);
    const source = await detectFace(sourcePath);
    const target = await detectFace(targetPath);
    const result = blendFace(source.canvas, source.detection, target.canvas, target.detection, strength);
    const ext = node_path_1.default.extname(outputPath).toLowerCase();
    const buffer = ext === ".jpg" || ext === ".jpeg"
        ? result.toBuffer("image/jpeg", { quality: 0.96 })
        : result.toBuffer("image/png");
    await promises_1.default.mkdir(node_path_1.default.dirname(outputPath), { recursive: true });
    await promises_1.default.writeFile(outputPath, buffer);
    console.log(`Face swap done: ${outputPath}`);
}
main().catch((err) => {
    console.error(`Failed: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
});
