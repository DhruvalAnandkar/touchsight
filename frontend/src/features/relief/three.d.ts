declare module "three" {
  export class Vector2 {
    constructor(x?: number, y?: number);
    x: number;
    y: number;
    set(x: number, y: number): this;
  }

  export class Vector3 {
    set(x: number, y: number, z: number): this;
    x: number;
    y: number;
    z: number;
  }

  export class Scene {
    background: unknown;
    add(...objects: object[]): this;
  }

  export class PerspectiveCamera {
    constructor(fov: number, aspect: number, near: number, far: number);
    aspect: number;
    near: number;
    far: number;
    position: Vector3;
    updateProjectionMatrix(): void;
  }

  export class WebGLRenderer {
    constructor(parameters?: { antialias?: boolean; alpha?: boolean });
    domElement: HTMLCanvasElement;
    setPixelRatio(value: number): void;
    setSize(width: number, height: number, updateStyle?: boolean): void;
    render(scene: Scene, camera: PerspectiveCamera): void;
    dispose(): void;
  }

  export class AmbientLight {
    constructor(color: number, intensity?: number);
  }

  export class DirectionalLight {
    constructor(color: number, intensity?: number);
    position: Vector3;
  }

  export class BufferGeometry {
    boundingSphere: { radius: number; center: Vector3 } | null;
    dispose(): void;
    computeVertexNormals(): void;
    center(): void;
    computeBoundingSphere(): void;
    getAttribute(name: string): {
      count: number;
      needsUpdate: boolean;
      getX(index: number): number;
      getY(index: number): number;
      setZ(index: number, value: number): void;
    };
  }

  export class PlaneGeometry extends BufferGeometry {
    constructor(width?: number, height?: number, widthSegments?: number, heightSegments?: number);
  }

  export class Material {
    dispose(): void;
  }

  export class Texture {
    needsUpdate: boolean;
    colorSpace: string;
    flipY: boolean;
    dispose(): void;
  }

  export class CanvasTexture extends Texture {
    constructor(canvas: HTMLCanvasElement);
  }

  export class TextureLoader {
    load(
      url: string,
      onLoad?: (texture: Texture) => void,
      onProgress?: (event: ProgressEvent) => void,
      onError?: (error: unknown) => void,
    ): Texture;
  }

  export const SRGBColorSpace: string;

  export class MeshStandardMaterial extends Material {
    constructor(parameters?: Record<string, unknown>);
    map: Texture | null;
    color: { set(hex: number): void };
    emissive: { set(hex: number): void };
    emissiveMap: Texture | null;
    emissiveIntensity: number;
    roughness: number;
    metalness: number;
    needsUpdate: boolean;
  }

  export class Mesh {
    constructor(geometry: BufferGeometry, material: Material);
    geometry: BufferGeometry;
    material: Material | Material[];
    rotation: { x: number; y: number; z: number };
    position: Vector3;
    visible: boolean;
  }

  export class Raycaster {
    constructor();
    setFromCamera(pointer: Vector2, camera: PerspectiveCamera): void;
    intersectObject(object: Mesh, recursive?: boolean): { uv?: Vector2 }[];
  }
}

declare module "three/addons/controls/OrbitControls.js" {
  import { PerspectiveCamera, Vector3 } from "three";

  export class OrbitControls {
    constructor(camera: PerspectiveCamera, domElement: HTMLElement);
    enableDamping: boolean;
    autoRotate: boolean;
    autoRotateSpeed: number;
    target: Vector3;
    update(): void;
    dispose(): void;
    addEventListener(type: "start" | "end" | "change", listener: () => void): void;
    removeEventListener(type: "start" | "end" | "change", listener: () => void): void;
  }
}

declare module "three/addons/loaders/STLLoader.js" {
  import { BufferGeometry } from "three";

  export class STLLoader {
    load(url: string, onLoad: (geometry: BufferGeometry) => void): void;
  }
}
