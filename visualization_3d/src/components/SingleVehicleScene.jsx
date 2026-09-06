import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { PerspectiveCamera, Sky, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import Road from './Road';
import GLTFCar from './GLTFCar';
import TrajectoryLine from './TrajectoryLine';
import { headingToThreeRotation } from '../utils/trajectoryUtils';

/**
 * SingleVehicleScene.jsx (Perfect Vehicle Positioning & Lane Offset)
 * 
 * Side 1 (isBaseline=true): Raw Baseline without GHOST AI (ISRO Raw Dataset / Phase 4 DR)
 * Side 2 (isBaseline=false): Final GHOST AI Tracker (Phase 7 Isotonic EKF, 2.40% Drift)
 */

function SingleVehicleController({ sample, allSamples, isBaseline = false }) {
  const cameraTargetRef = useRef(new THREE.Vector3(0, 0, 0));

  // Extract trajectory polyline points
  const points = useMemo(() => {
    if (!allSamples || allSamples.length === 0) return [];
    return allSamples.map((s) => {
      const trajData = isBaseline ? s.p4 : (s.p7 || s.p6);
      const headingRad = (trajData.h * Math.PI) / 180.0;
      // Lane offset 2.5m to the right of center line
      const offsetX = 2.5 * Math.cos(headingRad);
      const offsetZ = -2.5 * Math.sin(headingRad);

      return [
        trajData.x + offsetX,
        isBaseline ? 0.08 : 0.12,
        trajData.z + offsetZ
      ];
    });
  }, [allSamples, isBaseline]);

  useFrame((state) => {
    if (!sample) return;

    const trajData = isBaseline ? sample.p4 : (sample.p7 || sample.p6);
    const rawX = trajData.x;
    const rawZ = trajData.z;
    const headingDeg = trajData.h;
    const headingRad = (headingDeg * Math.PI) / 180.0;

    // Right lane offset (2.5m perpendicular to heading)
    const carX = rawX + 2.5 * Math.cos(headingRad);
    const carZ = rawZ - 2.5 * Math.sin(headingRad);

    const cam = state.camera;
    const targetCamX = carX;
    const targetCamY = 7.5;
    const targetCamZ = carZ - 15.0;

    cam.position.x = THREE.MathUtils.lerp(cam.position.x, targetCamX, 0.08);
    cam.position.y = THREE.MathUtils.lerp(cam.position.y, targetCamY, 0.08);
    cam.position.z = THREE.MathUtils.lerp(cam.position.z, targetCamZ, 0.08);

    cameraTargetRef.current.set(carX, 1.0, carZ + 6.0);
    cam.lookAt(cameraTargetRef.current);
  });

  if (!sample) return null;

  const isOutage = sample.state === 'OUTAGE';
  const trajData = isBaseline ? sample.p4 : (sample.p7 || sample.p6);
  const rawX = trajData.x;
  const rawZ = trajData.z;
  const heading = trajData.h;
  const headingRad = (heading * Math.PI) / 180.0;

  // Right lane offset position
  const carX = rawX + 2.5 * Math.cos(headingRad);
  const carZ = rawZ - 2.5 * Math.sin(headingRad);
  const pos = [carX, 0, carZ];

  const modelPath = isBaseline ? '/models/phase4_car.glb' : '/models/phase6_car.glb';
  const label = isBaseline ? 'RAW BASELINE (WITHOUT GHOST)' : 'GHOST AI TRACKER (FINAL)';
  const subLabel = isBaseline
    ? (isOutage ? 'GNSS LOST — 69.73% DRIFT' : 'GNSS AVAILABLE')
    : (isOutage ? 'GHOST ISOTONIC EKF — 2.40% DRIFT' : 'GNSS AVAILABLE');

  return (
    <>
      {/* 3D Trajectory Polyline */}
      <TrajectoryLine
        points={points}
        color={isBaseline ? '#DC2626' : '#10B981'}
        linewidth={2.2}
        opacity={1.0}
      />

      {/* 3D GLTF Car Model (Headlights facing 100% FORWARD, inside right lane) */}
      <GLTFCar
        modelPath={modelPath}
        position={pos}
        rotation={headingToThreeRotation(heading)}
        targetLength={isBaseline ? 4.5 : 4.8}
        rotationOffset={[0, 0, 0]}
        label={label}
        subLabel={subLabel}
        isBaseline={isBaseline}
      />
    </>
  );
}

export default function SingleVehicleScene({ sample, allSamples, gtRoadPoints = [], isBaseline = false }) {
  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', background: '#FCE4EC' }}>
      <Canvas shadows gl={{ antialias: true, alpha: false }}>
        {/* Explicit WebGL Clear Background Color (Soft Light Rose Pink) */}
        <color attach="background" args={["#FCE4EC"]} />

        {/* Soft Distance Horizon Fog */}
        <fog attach="fog" args={["#FCE4EC", 150, 3500]} />

        {/* Perspective Camera */}
        <PerspectiveCamera makeDefault fov={52} position={[0, 7.5, -15]} near={0.1} far={4000} />
        
        {/* Daylight Lighting */}
        <ambientLight intensity={0.9} />
        <directionalLight 
          castShadow 
          position={[50, 100, 50]} 
          intensity={1.6} 
          shadow-mapSize-width={2048} 
          shadow-mapSize-height={2048}
        />
        <hemisphereLight skyColor="#FFF0F3" groundColor="#E2A9BE" intensity={0.7} />

        {/* Daylight Sky */}
        <Sky sunPosition={[100, 80, 100]} inclination={0.5} azimuth={0.25} />

        {/* Dynamic Curved Highway Road & Green Grass Ribbon Landscape */}
        <Road roadPoints={gtRoadPoints} width={10} />

        {/* Single Vehicle & Trajectory Controller */}
        <SingleVehicleController sample={sample} allSamples={allSamples} isBaseline={isBaseline} />

        {/* Orbit Controls for manual inspection */}
        <OrbitControls 
          enablePan={true} 
          enableZoom={true} 
          maxPolarAngle={Math.PI / 2 - 0.05} 
        />
      </Canvas>
    </div>
  );
}
