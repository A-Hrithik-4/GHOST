import React, { useRef } from 'react';
import * as THREE from 'three';

/**
 * PlaceholderCar.jsx
 * 
 * Reusable vehicle component built from Three.js primitives.
 * 
 * Component Architecture:
 * Vehicle (State & Position Wrapper)
 *   └── VehicleModel (Visual Geometry)
 * 
 * Designed specifically so that VehicleModel can be swapped with an actual
 * .glb / .gltf car model in future steps without modifying the movement, 
 * heading, or camera tracking logic.
 */

function VehicleModel({ color = '#0284C7' }) {
  return (
    <group position={[0, 0.45, 0]}>
      {/* Main Car Body Chassis */}
      <mesh castShadow receiveShadow position={[0, 0.25, 0]}>
        <boxGeometry args={[1.8, 0.5, 3.8]} />
        <meshStandardMaterial 
          color={color} 
          metalness={0.7} 
          roughness={0.2} 
        />
      </mesh>

      {/* Cabin / Windshield */}
      <mesh castShadow position={[0, 0.65, -0.2]}>
        <boxGeometry args={[1.5, 0.45, 1.8]} />
        <meshStandardMaterial 
          color="#0F172A" 
          metalness={0.9} 
          roughness={0.1} 
          transparent 
          opacity={0.85} 
        />
      </mesh>

      {/* Front Headlights */}
      <mesh position={[-0.6, 0.25, 1.91]}>
        <boxGeometry args={[0.35, 0.15, 0.05]} />
        <meshStandardMaterial color="#E0F2FE" emissive="#38BDF8" emissiveIntensity={2} />
      </mesh>
      <mesh position={[0.6, 0.25, 1.91]}>
        <boxGeometry args={[0.35, 0.15, 0.05]} />
        <meshStandardMaterial color="#E0F2FE" emissive="#38BDF8" emissiveIntensity={2} />
      </mesh>

      {/* Rear Taillights */}
      <mesh position={[-0.6, 0.3, -1.91]}>
        <boxGeometry args={[0.35, 0.15, 0.05]} />
        <meshStandardMaterial color="#FEE2E2" emissive="#EF4444" emissiveIntensity={2} />
      </mesh>
      <mesh position={[0.6, 0.3, -1.91]}>
        <boxGeometry args={[0.35, 0.15, 0.05]} />
        <meshStandardMaterial color="#FEE2E2" emissive="#EF4444" emissiveIntensity={2} />
      </mesh>

      {/* 4 Wheels */}
      {/* Front Left */}
      <group position={[-0.95, 0.1, 1.1]} rotation={[0, 0, Math.PI / 2]}>
        <mesh castShadow>
          <cylinderGeometry args={[0.35, 0.35, 0.25, 24]} />
          <meshStandardMaterial color="#1E293B" roughness={0.8} />
        </mesh>
      </group>
      {/* Front Right */}
      <group position={[0.95, 0.1, 1.1]} rotation={[0, 0, Math.PI / 2]}>
        <mesh castShadow>
          <cylinderGeometry args={[0.35, 0.35, 0.25, 24]} />
          <meshStandardMaterial color="#1E293B" roughness={0.8} />
        </mesh>
      </group>
      {/* Rear Left */}
      <group position={[-0.95, 0.1, -1.1]} rotation={[0, 0, Math.PI / 2]}>
        <mesh castShadow>
          <cylinderGeometry args={[0.35, 0.35, 0.25, 24]} />
          <meshStandardMaterial color="#1E293B" roughness={0.8} />
        </mesh>
      </group>
      {/* Rear Right */}
      <group position={[0.95, 0.1, -1.1]} rotation={[0, 0, Math.PI / 2]}>
        <mesh castShadow>
          <cylinderGeometry args={[0.35, 0.35, 0.25, 24]} />
          <meshStandardMaterial color="#1E293B" roughness={0.8} />
        </mesh>
      </group>
    </group>
  );
}

export default function PlaceholderCar({ position = [0, 0, 0], rotation = [0, 0, 0], color = '#0284C7' }) {
  const groupRef = useRef();

  return (
    <group ref={groupRef} position={position} rotation={rotation}>
      <VehicleModel color={color} />
    </group>
  );
}
