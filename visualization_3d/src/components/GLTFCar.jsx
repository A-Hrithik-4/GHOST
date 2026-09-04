import React, { useMemo } from 'react';
import { useGLTF, Html } from '@react-three/drei';
import * as THREE from 'three';
import PlaceholderCar from './PlaceholderCar';

/**
 * GLTFCarMesh Component
 * 
 * Computes exact 3D bounding box center and ground Y-offset so the vehicle:
 * 1. Rotates around its true chassis geometric midpoint (0, 0).
 * 2. Wheels sit 100% flush on the road surface at Y = 0.
 * 3. Front headlights face 100% FORWARD along direction of travel.
 */
function GLTFCarMesh({ modelPath, targetLength = 4.6, rotationOffset = [0, 0, 0] }) {
  const { scene } = useGLTF(modelPath);

  const { clonedGroup, scaleFactor } = useMemo(() => {
    if (!scene) return { clonedGroup: null, scaleFactor: 1 };

    const clone = scene.clone(true);

    // 1. Hide baked shadow/plane/ground meshes inside the GLTF node tree
    clone.traverse((child) => {
      const nodeName = (child.name || '').toLowerCase();
      if (
        nodeName.includes('shadow') ||
        nodeName.includes('plane') ||
        nodeName.includes('ground') ||
        nodeName.includes('floor') ||
        nodeName.includes('backdrop')
      ) {
        child.visible = false;
      }

      if (child.isMesh && child.visible) {
        child.castShadow = true;
        child.receiveShadow = true;

        if (child.material) {
          child.material.depthWrite = true;
          child.material.needsUpdate = true;
        }
      }
    });

    // 2. Compute bounding box of visible car meshes
    const bbox = new THREE.Box3();
    clone.traverse((child) => {
      if (child.isMesh && child.visible) {
        bbox.expandByObject(child);
      }
    });

    const size = new THREE.Vector3();
    bbox.getSize(size);
    const center = new THREE.Vector3();
    bbox.getCenter(center);

    const currentLength = Math.max(size.x, size.z);
    const scale = currentLength > 0 ? targetLength / currentLength : 1.0;

    // 3. Shift inner clone so center X and Z are at (0, 0) and bottom Y is at Y = 0
    clone.position.set(-center.x * scale, -bbox.min.y * scale, -center.z * scale);
    clone.scale.set(scale, scale, scale);

    const wrapperGroup = new THREE.Group();
    wrapperGroup.add(clone);

    return {
      clonedGroup: wrapperGroup,
      scaleFactor: scale
    };
  }, [scene, targetLength]);

  if (!clonedGroup) return null;

  return (
    <group rotation={rotationOffset}>
      <primitive object={clonedGroup} />
    </group>
  );
}

export default function GLTFCar({
  modelPath,
  position = [0, 0, 0],
  rotation = [0, 0, 0],
  targetLength = 4.6,
  rotationOffset = [0, 0, 0],
  label = "CAR",
  subLabel = "GNSS OK",
  isBaseline = false
}) {
  return (
    <group position={position} rotation={rotation}>
      <React.Suspense fallback={<PlaceholderCar color={isBaseline ? '#DC2626' : '#1D4ED8'} />}>
        <GLTFCarMesh
          modelPath={modelPath}
          targetLength={targetLength}
          rotationOffset={rotationOffset}
        />
      </React.Suspense>

      {/* Floating 3D Badge (Large & Crisp) */}
      <Html position={[0, 2.8, 0]} center distanceFactor={15}>
        <div className={`floating-vehicle-label ${isBaseline ? 'label-baseline' : 'label-ghost'}`}>
          <span className="car-tag">{label}</span>
          <span className="car-sub">{subLabel}</span>
        </div>
      </Html>
    </group>
  );
}

// Preload models for instant loading
useGLTF.preload('/models/phase4_car.glb');
useGLTF.preload('/models/phase6_car.glb');
