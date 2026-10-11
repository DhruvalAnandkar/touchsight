declare module "three" {
  export class Scene {
    background: unknown;
    add(...objects: object[]): this;
  }

  export class Vector3 {
    set(x: number, y: number, z: number): this;
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
    boundingSphere: { radius: number } | null;
    dispose(): void;
    computeVertexNormals(): void;
    center(): void;
    computeBoundingSphere(): void;
  }

  export class Material {
    dispose(): void;
  }

  export class MeshStandardMaterial extends Material {
    constructor(parameters?: { color?: number; roughness?: number; metalness?: number });
  }

  export class Mesh {
    constructor(geometry: BufferGeometry, material: Material);
    geometry: BufferGeometry;
    material: Material | Material[];
    rotation: { x: number };
  }
}

declare module "three/addons/controls/OrbitControls.js" {
  import { PerspectiveCamera } from "three";

  export class OrbitControls {
    constructor(camera: PerspectiveCamera, domElement: HTMLElement);
    enableDamping: boolean;
    autoRotate: boolean;
    autoRotateSpeed: number;
    target: { set(x: number, y: number, z: number): void };
    update(): void;
    dispose(): void;
  }
}

declare module "three/addons/loaders/STLLoader.js" {
  import { BufferGeometry } from "three";

  export class STLLoader {
    load(url: string, onLoad: (geometry: BufferGeometry) => void): void;
  }
}
