import { useEffect, useRef, useState, type JSX, type KeyboardEvent } from "react";
import {
  AmbientLight,
  CanvasTexture,
  DirectionalLight,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  Raycaster,
  Scene,
  SRGBColorSpace,
  Texture,
  TextureLoader,
  Vector2,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { speak } from "../../voice/speak";

const PLANE_LONG_SIDE = 150;
const LONG_SEGMENTS = 256;
/** Brightest heightmap pixel, when the millimetre peak is not in the file. */
const PEAK_WORLD_UNITS = 8;

type Finish = "colored" | "print";

type GrayImage = {
  width: number;
  height: number;
  gray: Uint16Array;
};

type RegionPoint = {
  x: number;
  y: number;
  z: number;
};

export function ReliefView(props: {
  stlUrl: string | null;
  heightmapUrl?: string | null;
  regionIdUrl?: string | null;
  textureUrl?: string | null;
  regions?: { name: string; description: string }[];
}): JSX.Element {
  const stlUrl = props.stlUrl;
  const heightmapUrl = props.heightmapUrl ?? null;
  const regionIdUrl = props.regionIdUrl ?? null;
  const textureUrl = props.textureUrl ?? null;
  const mountRef = useRef<HTMLDivElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const regionsRef = useRef(props.regions ?? []);
  regionsRef.current = props.regions ?? [];
  const [finish, setFinish] = useState<Finish>("colored");
  const finishRef = useRef<Finish>(finish);
  finishRef.current = finish;
  const applyFinishRef = useRef<(mode: Finish) => void>(() => {});
  const viewerRef = useRef({
    cycle: (_delta: number) => {},
    speakCurrent: () => {},
  });

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    if (heightmapUrl) {
      return mountHeightmap({
        mount,
        frame: frameRef.current,
        heightmapUrl,
        regionIdUrl,
        textureUrl,
        regionsRef,
        finishRef,
        applyFinishRef,
        viewerRef,
      });
    }
    if (!stlUrl) return;
    return mountStl(mount, stlUrl);
  }, [stlUrl, heightmapUrl, regionIdUrl, textureUrl]);

  const showMesh = Boolean(stlUrl || heightmapUrl);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget) return;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      event.preventDefault();
      viewerRef.current.cycle(1);
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      event.preventDefault();
      viewerRef.current.cycle(-1);
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      viewerRef.current.speakCurrent();
    }
  };

  const choose = (mode: Finish) => {
    setFinish(mode);
    finishRef.current = mode;
    applyFinishRef.current(mode);
  };

  return (
    <div>
      {stlUrl ? (
        <a href={stlUrl} download="touchsight-relief.stl" className="font-bold text-gold underline">
          Download STL
        </a>
      ) : null}
      <div
        ref={frameRef}
        tabIndex={0}
        aria-label="3D relief"
        onKeyDown={onKeyDown}
        className="mt-3 rounded-2xl focus-visible:outline focus-visible:outline-offset-2"
      >
        <div
          className="relative w-full overflow-hidden rounded-2xl bg-desk-2"
          style={{ height: "24rem" }}
        >
          {heightmapUrl ? (
            <div className="absolute right-3 top-3 z-10 flex gap-2">
              <button
                type="button"
                aria-pressed={finish === "colored"}
                className={`rounded-full px-3 py-1 text-sm font-bold ${
                  finish === "colored" ? "bg-gold text-ink" : "bg-desk text-mist"
                }`}
                onClick={() => choose("colored")}
              >
                Colored
              </button>
              <button
                type="button"
                aria-pressed={finish === "print"}
                className={`rounded-full px-3 py-1 text-sm font-bold ${
                  finish === "print" ? "bg-gold text-ink" : "bg-desk text-mist"
                }`}
                onClick={() => choose("print")}
              >
                Print preview
              </button>
            </div>
          ) : null}
          {showMesh ? (
            <div ref={mountRef} className="h-full w-full" />
          ) : (
            <p className="flex h-full items-center justify-center px-6 text-center text-mist">
              The relief will appear here.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function mountStl(mount: HTMLDivElement, stlUrl: string): () => void {
  const scene = new Scene();
  scene.background = null;

  const camera = new PerspectiveCamera(40, 1, 0.1, 2000);
  camera.position.set(80, 70, 110);

  const renderer = new WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  mount.appendChild(renderer.domElement);
  renderer.domElement.style.display = "block";
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";

  scene.add(new AmbientLight(0xfff6e8, 0.72));
  const sun = new DirectionalLight(0xffffff, 0.95);
  sun.position.set(70, 120, 40);
  scene.add(sun);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  controls.autoRotateSpeed = 0.6;
  controls.target.set(0, 0, 0);

  let relief: Mesh | null = null;
  let alive = true;
  const loader = new STLLoader();
  loader.load(stlUrl, (geometry) => {
    if (!alive) {
      geometry.dispose();
      return;
    }
    geometry.computeVertexNormals();
    geometry.center();
    const material = new MeshStandardMaterial({
      color: 0xf4efe4,
      roughness: 0.82,
      metalness: 0,
    });
    relief = new Mesh(geometry, material);
    relief.rotation.x = -Math.PI / 2;
    scene.add(relief);

    geometry.computeBoundingSphere();
    const radius = geometry.boundingSphere?.radius ?? 60;
    camera.near = Math.max(radius / 100, 0.1);
    camera.far = radius * 20;
    camera.position.set(radius * 0.85, radius * 0.9, radius * 1.2);
    camera.updateProjectionMatrix();
    controls.update();
  });

  const fit = () => {
    const width = mount.clientWidth;
    const height = mount.clientHeight;
    if (width === 0 || height === 0) return;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  };
  fit();
  const observer = new ResizeObserver(fit);
  observer.observe(mount);

  let frame = 0;
  const animate = () => {
    frame = window.requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  };
  frame = window.requestAnimationFrame(animate);

  return () => {
    alive = false;
    window.cancelAnimationFrame(frame);
    observer.disconnect();
    controls.dispose();
    if (relief) {
      relief.geometry.dispose();
      const material = relief.material;
      if (!Array.isArray(material)) material.dispose();
    }
    renderer.dispose();
    renderer.domElement.remove();
  };
}

function mountHeightmap(args: {
  mount: HTMLDivElement;
  frame: HTMLDivElement | null;
  heightmapUrl: string;
  regionIdUrl: string | null;
  textureUrl: string | null;
  regionsRef: { current: { name: string; description: string }[] };
  finishRef: { current: Finish };
  applyFinishRef: { current: (mode: Finish) => void };
  viewerRef: { current: { cycle: (delta: number) => void; speakCurrent: () => void } };
}): () => void {
  const { mount, frame, heightmapUrl, regionIdUrl, textureUrl } = args;
  const scene = new Scene();
  scene.background = null;

  const camera = new PerspectiveCamera(40, 1, 0.1, 2000);
  camera.position.set(80, 70, 110);

  const renderer = new WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  mount.appendChild(renderer.domElement);
  renderer.domElement.style.display = "block";
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";

  scene.add(new AmbientLight(0xfff4e4, 0.8));
  const sun = new DirectionalLight(0xfffaf2, 0.58);
  sun.position.set(46, 120, 36);
  scene.add(sun);
  const fill = new DirectionalLight(0xd5e2f2, 0.36);
  fill.position.set(-72, 48, -28);
  scene.add(fill);

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = !reducedMotion;
  controls.autoRotateSpeed = 0.6;
  controls.target.set(0, 0, 0);
  const stopSpin = () => {
    controls.autoRotate = false;
  };
  const resumeSpin = () => {
    controls.autoRotate = !reducedMotion;
  };
  controls.addEventListener("start", stopSpin);
  controls.addEventListener("end", resumeSpin);

  let relief: Mesh | null = null;
  let glow: Mesh | null = null;
  let colorTexture: Texture | null = null;
  let glowTexture: CanvasTexture | null = null;
  let glowCanvas: HTMLCanvasElement | null = null;
  let material: MeshStandardMaterial | null = null;
  let alive = true;
  let selected = -1;
  const discovered: number[] = [];
  const centroids = new Map<number, RegionPoint>();
  const raycaster = new Raycaster();
  const pointer = new Vector2();
  let regionImage: GrayImage | null = null;
  let downX = 0;
  let downY = 0;

  const paintGlow = (regionValue: number) => {
    if (!glowTexture || !glowCanvas || !regionImage) return;
    const context = glowCanvas.getContext("2d");
    if (!context) return;
    const pixels = context.createImageData(glowCanvas.width, glowCanvas.height);
    const rgba = pixels.data;
    for (let i = 0; i < regionImage.gray.length; i += 1) {
      const on = regionImage.gray[i] === regionValue;
      const offset = i * 4;
      rgba[offset] = on ? 255 : 0;
      rgba[offset + 1] = on ? 255 : 0;
      rgba[offset + 2] = on ? 255 : 0;
      rgba[offset + 3] = 255;
    }
    context.putImageData(pixels, 0, 0);
    glowTexture.needsUpdate = true;
    if (glow) glow.visible = regionValue > 0;
  };

  const lookAtRegion = (index: number) => {
    const point = centroids.get(index);
    if (!point) return;
    controls.target.set(point.x, point.y, point.z);
    controls.update();
  };

  const speakIndex = (index: number) => {
    const region = args.regionsRef.current[index];
    if (!region) return;
    const text = `${region.name}. ${region.description}`.trim();
    if (!text || text === ".") return;
    speak(text, { priority: true });
  };

  const focusRegion = (index: number, announce: boolean) => {
    selected = index;
    paintGlow(index + 1);
    lookAtRegion(index);
    if (announce) speakIndex(index);
  };

  const indexes = () => {
    const named = args.regionsRef.current.length;
    if (named > 0) return Array.from({ length: named }, (_, index) => index);
    return discovered;
  };

  args.viewerRef.current = {
    cycle: (delta: number) => {
      const order = indexes();
      if (order.length === 0) return;
      const place = order.indexOf(selected);
      const next =
        place < 0
          ? delta > 0
            ? 0
            : order.length - 1
          : (place + delta + order.length) % order.length;
      focusRegion(order[next], false);
    },
    speakCurrent: () => {
      if (selected >= 0) speakIndex(selected);
    },
  };

  const applyFinish = (mode: Finish) => {
    if (!material) return;
    if (mode === "print") {
      material.map = null;
      material.color.set(0xf4efe4);
      material.roughness = 0.82;
    } else {
      material.map = colorTexture;
      material.color.set(0xffffff);
      material.roughness = 0.62;
    }
    material.needsUpdate = true;
  };
  args.applyFinishRef.current = applyFinish;

  const onPointerDown = (event: PointerEvent) => {
    downX = event.clientX;
    downY = event.clientY;
  };

  const onPointerUp = (event: PointerEvent) => {
    if (event.button !== 0 || !relief || !regionImage) return;
    if (Math.hypot(event.clientX - downX, event.clientY - downY) > 8) return;
    const rect = renderer.domElement.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    pointer.set(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObject(relief, false)[0];
    if (!hit?.uv) return;
    const x = Math.round(hit.uv.x * (regionImage.width - 1));
    const y = Math.round((1 - hit.uv.y) * (regionImage.height - 1));
    if (x < 0 || y < 0 || x >= regionImage.width || y >= regionImage.height) return;
    const value = regionImage.gray[y * regionImage.width + x];
    if (value <= 0) return;
    frame?.focus();
    focusRegion(value - 1, true);
  };

  renderer.domElement.addEventListener("pointerdown", onPointerDown);
  renderer.domElement.addEventListener("pointerup", onPointerUp);

  const fit = () => {
    const width = mount.clientWidth;
    const height = mount.clientHeight;
    if (width === 0 || height === 0) return;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  };
  fit();
  const observer = new ResizeObserver(fit);
  observer.observe(mount);

  let frameId = 0;
  const animate = () => {
    frameId = window.requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  };
  frameId = window.requestAnimationFrame(animate);

  const build = async () => {
    try {
      const heightImage = await loadGray(heightmapUrl);
      if (!alive) return;
      const regions = regionIdUrl ? await loadGray(regionIdUrl) : null;
      if (!alive) return;
      if (textureUrl) {
        colorTexture = await loadColorTexture(textureUrl);
        if (!alive) {
          colorTexture?.dispose();
          colorTexture = null;
          return;
        }
      }
      regionImage = regions;
      const { planeW, planeH, segX, segY } = planeLayout(heightImage.width, heightImage.height);
      const peak = maxSample(heightImage.gray);
      const geometry = new PlaneGeometry(planeW, planeH, segX, segY);
      const position = geometry.getAttribute("position");
      const columns = segX + 1;
      for (let index = 0; index < position.count; index += 1) {
        const ix = index % columns;
        const iy = Math.floor(index / columns);
        const u = ix / segX;
        const v = 1 - iy / segY;
        const sample = bilinear(
          heightImage,
          u * (heightImage.width - 1),
          (1 - v) * (heightImage.height - 1),
        );
        position.setZ(index, (sample / peak) * PEAK_WORLD_UNITS);
      }
      position.needsUpdate = true;
      geometry.computeVertexNormals();

      material = new MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.62,
        metalness: 0,
        map: colorTexture,
      });
      relief = new Mesh(geometry, material);
      relief.rotation.x = -Math.PI / 2;
      scene.add(relief);

      if (regions) {
        glowCanvas = document.createElement("canvas");
        glowCanvas.width = regions.width;
        glowCanvas.height = regions.height;
        glowTexture = new CanvasTexture(glowCanvas);
        const glowMaterial = new MeshStandardMaterial({
          color: 0x000000,
          emissive: 0xf0c14b,
          emissiveMap: glowTexture,
          emissiveIntensity: 1.35,
          roughness: 1,
          metalness: 0,
          transparent: true,
          depthWrite: false,
          polygonOffset: true,
          polygonOffsetFactor: -2,
          polygonOffsetUnits: -2,
        });
        glow = new Mesh(geometry, glowMaterial);
        glow.rotation.x = -Math.PI / 2;
        glow.position.y = 0.08;
        glow.visible = false;
        scene.add(glow);
        collectRegions(regions, heightImage, planeW, planeH, peak, centroids, discovered);
      }

      geometry.computeBoundingSphere();
      const radius = geometry.boundingSphere?.radius ?? Math.max(planeW, planeH) * 0.6;
      const center = geometry.boundingSphere?.center;
      camera.near = Math.max(radius / 100, 0.1);
      camera.far = radius * 20;
      camera.position.set(radius * 0.85, radius * 0.95, radius * 1.25);
      if (center) controls.target.set(center.x, center.z, -center.y);
      camera.updateProjectionMatrix();
      controls.update();
      applyFinish(args.finishRef.current);
      if (selected >= 0) focusRegion(selected, false);
    } catch {
      // A bad heightmap leaves the lit stage empty; the STL link still works.
    }
  };
  void build();

  return () => {
    alive = false;
    args.viewerRef.current = { cycle: () => {}, speakCurrent: () => {} };
    args.applyFinishRef.current = () => {};
    window.cancelAnimationFrame(frameId);
    observer.disconnect();
    renderer.domElement.removeEventListener("pointerdown", onPointerDown);
    renderer.domElement.removeEventListener("pointerup", onPointerUp);
    controls.removeEventListener("start", stopSpin);
    controls.removeEventListener("end", resumeSpin);
    controls.dispose();
    colorTexture?.dispose();
    glowTexture?.dispose();
    relief?.geometry.dispose();
    disposeMaterial(relief);
    disposeMaterial(glow);
    renderer.dispose();
    renderer.domElement.remove();
  };
}

function disposeMaterial(mesh: Mesh | null): void {
  if (!mesh) return;
  const material = mesh.material;
  if (!Array.isArray(material)) material.dispose();
}

function planeLayout(width: number, height: number): {
  planeW: number;
  planeH: number;
  segX: number;
  segY: number;
} {
  if (width >= height) {
    return {
      planeW: PLANE_LONG_SIDE,
      planeH: PLANE_LONG_SIDE * (height / width),
      segX: LONG_SEGMENTS,
      segY: Math.max(1, Math.round((LONG_SEGMENTS * height) / width)),
    };
  }
  return {
    planeH: PLANE_LONG_SIDE,
    planeW: PLANE_LONG_SIDE * (width / height),
    segY: LONG_SEGMENTS,
    segX: Math.max(1, Math.round((LONG_SEGMENTS * width) / height)),
  };
}

function maxSample(values: Uint16Array): number {
  let peak = 1;
  for (let index = 0; index < values.length; index += 1) {
    if (values[index] > peak) peak = values[index];
  }
  return peak;
}

function bilinear(image: GrayImage, x: number, y: number): number {
  const x0 = Math.max(0, Math.min(image.width - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(image.height - 1, Math.floor(y)));
  const x1 = Math.min(image.width - 1, x0 + 1);
  const y1 = Math.min(image.height - 1, y0 + 1);
  const tx = Math.max(0, Math.min(1, x - x0));
  const ty = Math.max(0, Math.min(1, y - y0));
  const row0 = y0 * image.width;
  const row1 = y1 * image.width;
  const top = image.gray[row0 + x0] * (1 - tx) + image.gray[row0 + x1] * tx;
  const bottom = image.gray[row1 + x0] * (1 - tx) + image.gray[row1 + x1] * tx;
  return top * (1 - ty) + bottom * ty;
}

function collectRegions(
  regions: GrayImage,
  heights: GrayImage,
  planeW: number,
  planeH: number,
  peak: number,
  centroids: Map<number, RegionPoint>,
  discovered: number[],
): void {
  const sums = new Map<number, { x: number; y: number; n: number }>();
  for (let index = 0; index < regions.gray.length; index += 1) {
    const value = regions.gray[index];
    if (value <= 0) continue;
    const bucket = sums.get(value) ?? { x: 0, y: 0, n: 0 };
    bucket.x += index % regions.width;
    bucket.y += Math.floor(index / regions.width);
    bucket.n += 1;
    sums.set(value, bucket);
  }
  const ids = Array.from(sums.keys()).sort((a, b) => a - b);
  for (const value of ids) {
    const bucket = sums.get(value);
    if (!bucket || bucket.n === 0) continue;
    const px = bucket.x / bucket.n;
    const py = bucket.y / bucket.n;
    const u = regions.width <= 1 ? 0.5 : px / (regions.width - 1);
    const v = regions.height <= 1 ? 0.5 : 1 - py / (regions.height - 1);
    const localX = (u - 0.5) * planeW;
    const localY = (v - 0.5) * planeH;
    const sample = bilinear(
      heights,
      u * (heights.width - 1),
      (1 - v) * (heights.height - 1),
    );
    const worldY = (sample / peak) * PEAK_WORLD_UNITS;
    centroids.set(value - 1, { x: localX, y: worldY, z: -localY });
    discovered.push(value - 1);
  }
}

function loadColorTexture(url: string): Promise<Texture | null> {
  return new Promise((resolve) => {
    const loader = new TextureLoader();
    loader.load(
      url,
      (texture) => {
        texture.colorSpace = SRGBColorSpace;
        resolve(texture);
      },
      undefined,
      () => resolve(null),
    );
  });
}

async function loadGray(url: string): Promise<GrayImage> {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error("image fetch failed");
    const bytes = new Uint8Array(await response.arrayBuffer());
    return await decodePngGray(bytes);
  } catch {
    return decodeGrayFromCanvas(url);
  }
}

function decodeGrayFromCanvas(url: string): Promise<GrayImage> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (!context || canvas.width < 2 || canvas.height < 2) {
        reject(new Error("image canvas failed"));
        return;
      }
      context.drawImage(image, 0, 0);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
      const gray = new Uint16Array(canvas.width * canvas.height);
      for (let index = 0, pixel = 0; index < gray.length; index += 1, pixel += 4) {
        gray[index] = Math.round(
          0.2126 * pixels[pixel] + 0.7152 * pixels[pixel + 1] + 0.0722 * pixels[pixel + 2],
        );
      }
      resolve({ width: canvas.width, height: canvas.height, gray });
    };
    image.onerror = () => reject(new Error("image failed"));
    image.src = url;
  });
}

async function decodePngGray(bytes: Uint8Array): Promise<GrayImage> {
  const signature = [137, 80, 78, 71, 13, 10, 26, 10];
  if (bytes.length < 8 || signature.some((value, index) => bytes[index] !== value)) {
    throw new Error("not a png");
  }
  let offset = 8;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = 0;
  const idat: Uint8Array[] = [];
  while (offset + 8 <= bytes.length) {
    const length = readUint32(bytes, offset);
    const type = String.fromCharCode(
      bytes[offset + 4],
      bytes[offset + 5],
      bytes[offset + 6],
      bytes[offset + 7],
    );
    const start = offset + 8;
    const end = start + length;
    if (end + 4 > bytes.length) throw new Error("truncated png");
    if (type === "IHDR") {
      width = readUint32(bytes, start);
      height = readUint32(bytes, start + 4);
      bitDepth = bytes[start + 8];
      colorType = bytes[start + 9];
      if (bytes[start + 12] !== 0) throw new Error("interlaced png");
    } else if (type === "IDAT") {
      idat.push(bytes.subarray(start, end));
    } else if (type === "IEND") {
      break;
    }
    offset = end + 4;
  }
  if (width < 2 || height < 2) throw new Error("png too small");
  const payload = concatBytes(idat);
  const inflated = await inflateZlib(payload);
  const channels = pngChannels(colorType);
  if (bitDepth < 8 || bitDepth % 8 !== 0 || channels === 0) throw new Error("unsupported png");
  const bytesPerPixel = channels * (bitDepth / 8);
  if (inflated.length < height * (1 + width * bytesPerPixel)) throw new Error("short png");
  const raw = unfilterPng(inflated, width, height, bytesPerPixel);
  return { width, height, gray: graySamples(raw, width, height, colorType, bitDepth) };
}

function pngChannels(colorType: number): number {
  if (colorType === 0) return 1;
  if (colorType === 2) return 3;
  if (colorType === 4) return 2;
  if (colorType === 6) return 4;
  return 0;
}

function readUint32(bytes: Uint8Array, offset: number): number {
  return (
    (bytes[offset] << 24) |
    (bytes[offset + 1] << 16) |
    (bytes[offset + 2] << 8) |
    bytes[offset + 3]
  ) >>> 0;
}

function concatBytes(parts: Uint8Array[]): Uint8Array {
  let length = 0;
  for (const part of parts) length += part.length;
  const out = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

async function inflateZlib(data: Uint8Array): Promise<Uint8Array> {
  const buffer = new ArrayBuffer(data.byteLength);
  new Uint8Array(buffer).set(data);
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function unfilterPng(inflated: Uint8Array, width: number, height: number, bytesPerPixel: number): Uint8Array {
  const stride = width * bytesPerPixel;
  const out = new Uint8Array(height * stride);
  let source = 0;
  for (let y = 0; y < height; y += 1) {
    const filter = inflated[source];
    source += 1;
    const row = y * stride;
    for (let x = 0; x < stride; x += 1) {
      const raw = inflated[source];
      source += 1;
      const left = x >= bytesPerPixel ? out[row + x - bytesPerPixel] : 0;
      const up = y > 0 ? out[row - stride + x] : 0;
      const upLeft = y > 0 && x >= bytesPerPixel ? out[row - stride + x - bytesPerPixel] : 0;
      let value = raw;
      if (filter === 1) value = (raw + left) & 255;
      else if (filter === 2) value = (raw + up) & 255;
      else if (filter === 3) value = (raw + Math.floor((left + up) / 2)) & 255;
      else if (filter === 4) value = (raw + paeth(left, up, upLeft)) & 255;
      else if (filter !== 0) throw new Error("bad png filter");
      out[row + x] = value;
    }
  }
  return out;
}

function paeth(left: number, up: number, upLeft: number): number {
  const estimate = left + up - upLeft;
  const leftDistance = Math.abs(estimate - left);
  const upDistance = Math.abs(estimate - up);
  const diagonal = Math.abs(estimate - upLeft);
  if (leftDistance <= upDistance && leftDistance <= diagonal) return left;
  if (upDistance <= diagonal) return up;
  return upLeft;
}

function graySamples(
  raw: Uint8Array,
  width: number,
  height: number,
  colorType: number,
  bitDepth: number,
): Uint16Array {
  const count = width * height;
  const out = new Uint16Array(count);
  if (colorType === 0 && bitDepth === 16) {
    for (let index = 0, pixel = 0; index < count; index += 1, pixel += 2) {
      out[index] = (raw[pixel] << 8) | raw[pixel + 1];
    }
    return out;
  }
  if (colorType === 0 && bitDepth === 8) {
    for (let index = 0; index < count; index += 1) out[index] = raw[index];
    return out;
  }
  const bytesPerPixel = pngChannels(colorType) * (bitDepth / 8);
  for (let index = 0; index < count; index += 1) {
    const pixel = index * bytesPerPixel;
    if (bitDepth === 16) {
      const red = (raw[pixel] << 8) | raw[pixel + 1];
      const green = colorType === 2 || colorType === 6 ? (raw[pixel + 2] << 8) | raw[pixel + 3] : red;
      const blue = colorType === 2 || colorType === 6 ? (raw[pixel + 4] << 8) | raw[pixel + 5] : red;
      out[index] = Math.round(0.2126 * red + 0.7152 * green + 0.0722 * blue);
    } else {
      const red = raw[pixel];
      const green = colorType === 2 || colorType === 6 ? raw[pixel + 1] : red;
      const blue = colorType === 2 || colorType === 6 ? raw[pixel + 2] : red;
      out[index] = Math.round(0.2126 * red + 0.7152 * green + 0.0722 * blue);
    }
  }
  return out;
}
