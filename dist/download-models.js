"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const promises_1 = __importDefault(require("node:fs/promises"));
const node_path_1 = __importDefault(require("node:path"));
const BASE_URLS = [
    "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model",
    "https://unpkg.com/@vladmandic/face-api/model",
    "https://raw.githubusercontent.com/vladmandic/face-api/master/model"
];
const MODELS = ["tiny_face_detector_model", "face_landmark_68_tiny_model"];
async function downloadFile(urls, toPath) {
    let lastError = "unknown error";
    for (const url of urls) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                lastError = `${url} (${response.status})`;
                continue;
            }
            const data = Buffer.from(await response.arrayBuffer());
            await promises_1.default.mkdir(node_path_1.default.dirname(toPath), { recursive: true });
            await promises_1.default.writeFile(toPath, data);
            return;
        }
        catch (err) {
            lastError = `${url} (${err instanceof Error ? err.message : String(err)})`;
        }
    }
    throw new Error(`下载失败: ${lastError}`);
}
async function main() {
    const modelDir = node_path_1.default.resolve(process.cwd(), "models");
    await promises_1.default.mkdir(modelDir, { recursive: true });
    for (const modelName of MODELS) {
        const manifestName = `${modelName}-weights_manifest.json`;
        const manifestUrls = BASE_URLS.map((base) => `${base}/${manifestName}`);
        const manifestPath = node_path_1.default.join(modelDir, manifestName);
        console.log(`下载 ${manifestName} ...`);
        await downloadFile(manifestUrls, manifestPath);
        const manifestRaw = await promises_1.default.readFile(manifestPath, "utf8");
        const manifest = JSON.parse(manifestRaw);
        for (const group of manifest) {
            for (const shardName of group.paths) {
                const shardUrls = BASE_URLS.map((base) => `${base}/${shardName}`);
                const shardPath = node_path_1.default.join(modelDir, shardName);
                console.log(`下载 ${shardName} ...`);
                await downloadFile(shardUrls, shardPath);
            }
        }
    }
    console.log(`模型下载完成: ${modelDir}`);
}
main().catch((err) => {
    console.error(`下载失败: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
});
