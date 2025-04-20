import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

interface SystemNode {
  id: string;
  name: string;
  position: [number, number, number];
  color: string;
}

interface DataFlowProps {
  nodes: SystemNode[];
  width?: number;
  height?: number;
  className?: string;
}

export function DataFlowViz({ nodes, width = 800, height = 400, className }: DataFlowProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene>();
  const cameraRef = useRef<THREE.PerspectiveCamera>();
  const rendererRef = useRef<THREE.WebGLRenderer>();
  const controlsRef = useRef<OrbitControls>();
  const particlesRef = useRef<THREE.Points[]>([]);

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
    sceneRef.current = scene;

    // Initialize camera
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = 5;
    cameraRef.current = camera;

    // Initialize renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Add controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controlsRef.current = controls;

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 1, 0);
    scene.add(directionalLight);

    // Create nodes
    nodes.forEach(node => {
      const geometry = new THREE.SphereGeometry(0.2, 32, 32);
      const material = new THREE.MeshPhongMaterial({ 
        color: node.color,
        emissive: node.color,
        emissiveIntensity: 0.5
      });
      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(...node.position);
      scene.add(sphere);

      // Add text label
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      if (context) {
        context.font = '24px Arial';
        context.fillStyle = 'white';
        context.fillText(node.name, 0, 24);
        const texture = new THREE.CanvasTexture(canvas);
        const labelMaterial = new THREE.SpriteMaterial({ map: texture });
        const label = new THREE.Sprite(labelMaterial);
        label.position.set(node.position[0], node.position[1] + 0.4, node.position[2]);
        label.scale.set(1, 0.5, 1);
        scene.add(label);
      }
    });

    // Create particle systems for data flow
    nodes.forEach((node, i) => {
      if (i < nodes.length - 1) {
        const nextNode = nodes[i + 1];
        const particleCount = 50;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);

        for (let j = 0; j < particleCount; j++) {
          positions[j * 3] = node.position[0];
          positions[j * 3 + 1] = node.position[1];
          positions[j * 3 + 2] = node.position[2];

          const color = new THREE.Color(node.color);
          colors[j * 3] = color.r;
          colors[j * 3 + 1] = color.g;
          colors[j * 3 + 2] = color.b;
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
          size: 0.05,
          vertexColors: true,
          transparent: true,
          opacity: 0.8
        });

        const particles = new THREE.Points(geometry, material);
        particlesRef.current.push(particles);
        scene.add(particles);
      }
    });

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();

      // Animate particles
      particlesRef.current.forEach((particles, i) => {
        const positions = particles.geometry.attributes.position.array as Float32Array;
        const node = nodes[i];
        const nextNode = nodes[i + 1];

        for (let j = 0; j < positions.length; j += 3) {
          const progress = (Math.sin(Date.now() * 0.001 + j * 0.1) + 1) / 2;
          positions[j] = node.position[0] + (nextNode.position[0] - node.position[0]) * progress;
          positions[j + 1] = node.position[1] + (nextNode.position[1] - node.position[1]) * progress;
          positions[j + 2] = node.position[2] + (nextNode.position[2] - node.position[2]) * progress;
        }
        particles.geometry.attributes.position.needsUpdate = true;
      });

      renderer.render(scene, camera);
    };
    animate();

    // Cleanup
    return () => {
      if (containerRef.current && renderer.domElement) {
        containerRef.current.removeChild(renderer.domElement);
      }
      particlesRef.current.forEach(particles => {
        particles.geometry.dispose();
        (particles.material as THREE.Material).dispose();
      });
    };
  }, [nodes, width, height]);

  return (
    <div 
      ref={containerRef} 
      className={`rounded-lg overflow-hidden bg-black/20 backdrop-blur-sm ${className}`}
      style={{ width, height }}
    />
  );
} 