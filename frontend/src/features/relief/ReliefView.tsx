import { useEffect, useRef, type JSX } from "react";
import {
  AmbientLight,
  DirectionalLight,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  Scene,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

export function ReliefView(props: { stlUrl: string | null }): JSX.Element {
  const stlUrl = props.stlUrl;
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !stlUrl) return;

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
  }, [stlUrl]);

  return (
    <div>
      {stlUrl ? (
        <a href={stlUrl} download="touchsight-relief.stl" className="font-bold text-gold underline">
          Download STL
        </a>
      ) : null}
      <div
        className="relative mt-3 w-full overflow-hidden rounded-2xl bg-desk-2"
        style={{ height: "24rem" }}
      >
        {stlUrl ? (
          <div ref={mountRef} className="h-full w-full" />
        ) : (
          <p className="flex h-full items-center justify-center px-6 text-center text-mist">
            The relief will appear here.
          </p>
        )}
      </div>
    </div>
  );
}
