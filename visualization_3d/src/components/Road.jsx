import React, { useMemo } from 'react';
import * as THREE from 'three';

/**
 * Road.jsx (Guaranteed Visible Grass Strips & Bounded Highway Ribbon)
 * 
 * Extrudes a 10-meter wide asphalt road ribbon along the Ground Truth trajectory,
 * flanked by 60-meter wide vibrant green grass ribbons on BOTH left and right sides.
 */

export default function Road({ roadPoints = [], width = 10 }) {
  const {
    asphaltGeo,
    yellowLineGeo,
    whiteLeftLineGeo,
    whiteRightLineGeo,
    curbLeftGeo,
    curbRightGeo,
    grassLeftGeo,
    grassRightGeo,
    centerPos
  } = useMemo(() => {
    if (!roadPoints || roadPoints.length < 2) {
      const dummyPts = [new THREE.Vector3(0, 0, -500), new THREE.Vector3(0, 0, 1500)];
      const curve = new THREE.CatmullRomCurve3(dummyPts);
      const points = curve.getPoints(200);
      return buildRoadGeometries(points, width);
    }

    const vecPoints = [];
    const step = Math.max(1, Math.floor(roadPoints.length / 400));
    for (let i = 0; i < roadPoints.length; i += step) {
      const pt = roadPoints[i];
      vecPoints.push(new THREE.Vector3(pt[0], 0.02, pt[1]));
    }
    const lastPt = roadPoints[roadPoints.length - 1];
    vecPoints.push(new THREE.Vector3(lastPt[0], 0.02, lastPt[1]));

    const curve = new THREE.CatmullRomCurve3(vecPoints, false, 'centripetal', 0.5);
    const sampledPoints = curve.getPoints(1200);

    return buildRoadGeometries(sampledPoints, width);
  }, [roadPoints, width]);

  return (
    <group>
      {/* 1. Large Base Ground Plane */}
      <mesh receiveShadow rotation={[-Math.PI / 2, 0, 0]} position={[centerPos.x, -0.05, centerPos.z]}>
        <planeGeometry args={[4000, 4500]} />
        <meshStandardMaterial color="#16A34A" roughness={0.95} metalness={0.02} side={THREE.DoubleSide} />
      </mesh>

      {/* 2. Left Vibrant Green Grass Ribbon (Width 80m) */}
      {grassLeftGeo && (
        <mesh receiveShadow geometry={grassLeftGeo}>
          <meshStandardMaterial color="#15803D" roughness={0.9} emissive="#064E3B" emissiveIntensity={0.15} side={THREE.DoubleSide} />
        </mesh>
      )}

      {/* 3. Right Vibrant Green Grass Ribbon (Width 80m) */}
      {grassRightGeo && (
        <mesh receiveShadow geometry={grassRightGeo}>
          <meshStandardMaterial color="#15803D" roughness={0.9} emissive="#064E3B" emissiveIntensity={0.15} side={THREE.DoubleSide} />
        </mesh>
      )}

      {/* 4. Slate Asphalt Highway Ribbon */}
      {asphaltGeo && (
        <mesh receiveShadow geometry={asphaltGeo}>
          <meshStandardMaterial color="#334155" roughness={0.5} metalness={0.15} side={THREE.DoubleSide} />
        </mesh>
      )}

      {/* 5. Double Yellow Center Line */}
      {yellowLineGeo && (
        <mesh geometry={yellowLineGeo}>
          <meshStandardMaterial color="#F59E0B" roughness={0.3} emissive="#D97706" emissiveIntensity={0.3} side={THREE.DoubleSide} />
        </mesh>
      )}

      {/* 6. Solid White Edge Lines */}
      {whiteLeftLineGeo && (
        <mesh geometry={whiteLeftLineGeo}>
          <meshStandardMaterial color="#FFFFFF" roughness={0.3} side={THREE.DoubleSide} />
        </mesh>
      )}
      {whiteRightLineGeo && (
        <mesh geometry={whiteRightLineGeo}>
          <meshStandardMaterial color="#FFFFFF" roughness={0.3} side={THREE.DoubleSide} />
        </mesh>
      )}

      {/* 7. Concrete Curbs */}
      {curbLeftGeo && (
        <mesh geometry={curbLeftGeo}>
          <meshStandardMaterial color="#94A3B8" roughness={0.8} side={THREE.DoubleSide} />
        </mesh>
      )}
      {curbRightGeo && (
        <mesh geometry={curbRightGeo}>
          <meshStandardMaterial color="#94A3B8" roughness={0.8} side={THREE.DoubleSide} />
        </mesh>
      )}
    </group>
  );
}

function buildRoadGeometries(points, width) {
  if (points.length < 2) return {};

  const numPoints = points.length;
  
  let sumX = 0, sumZ = 0;
  for (let pt of points) { sumX += pt.x; sumZ += pt.z; }
  const centerPos = { x: sumX / numPoints, z: sumZ / numPoints };

  const halfW = width / 2;
  const grassW = 80; // 80 meters of grass on left and right sides of road

  const aspVerts = [], aspNorms = [], aspUVs = [], aspIndices = [];
  const yelVerts = [], yelIndices = [];
  const wLVerts = [], wLIndices = [];
  const wRVerts = [], wRIndices = [];
  const cLVerts = [], cLIndices = [];
  const cRVerts = [], cRIndices = [];
  const gLVerts = [], gLIndices = [];
  const gRVerts = [], gRIndices = [];

  for (let i = 0; i < numPoints; i++) {
    const p = points[i];
    
    let tangent = new THREE.Vector3();
    if (i < numPoints - 1) {
      tangent.subVectors(points[i + 1], p);
    } else {
      tangent.subVectors(p, points[i - 1]);
    }
    tangent.y = 0;
    tangent.normalize();

    const normal = new THREE.Vector3(-tangent.z, 0, tangent.x).normalize();

    const leftP = new THREE.Vector3().addVectors(p, normal.clone().multiplyScalar(-halfW));
    const rightP = new THREE.Vector3().addVectors(p, normal.clone().multiplyScalar(halfW));

    // 1. Asphalt Ribbon
    aspVerts.push(leftP.x, p.y, leftP.z);
    aspVerts.push(rightP.x, p.y, rightP.z);
    aspNorms.push(0, 1, 0, 0, 1, 0);
    aspUVs.push(0, i / numPoints, 1, i / numPoints);

    // 2. Yellow Center Line
    const yL = new THREE.Vector3().addVectors(p, normal.clone().multiplyScalar(-0.2));
    const yR = new THREE.Vector3().addVectors(p, normal.clone().multiplyScalar(0.2));
    yelVerts.push(yL.x, p.y + 0.005, yL.z);
    yelVerts.push(yR.x, p.y + 0.005, yR.z);

    // 3. White Left Line
    const wL1 = new THREE.Vector3().addVectors(p, normal.clone().multiplyScalar(-halfW + 0.35));
    const wL2 = new THREE.Vector3().addVectors(p, normal.clone().multiplyScalar(-halfW + 0.65));
    wLVerts.push(wL1.x, p.y + 0.004, wL1.z);
    wLVerts.push(wL2.x, p.y + 0.004, wL2.z);

    // 4. White Right Line
    const wR1 = new THREE.Vector3().addVectors(p, normal.clone().multiplyScalar(halfW - 0.65));
    const wR2 = new THREE.Vector3().addVectors(p, normal.clone().multiplyScalar(halfW - 0.35));
    wRVerts.push(wR1.x, p.y + 0.004, wR1.z);
    wRVerts.push(wR2.x, p.y + 0.004, wR2.z);

    // 5. Left Curb
    const cL1 = new THREE.Vector3().addVectors(p, normal.clone().multiplyScalar(-halfW - 0.6));
    const cL2 = new THREE.Vector3().addVectors(p, normal.clone().multiplyScalar(-halfW));
    cLVerts.push(cL1.x, p.y + 0.01, cL1.z);
    cLVerts.push(cL2.x, p.y + 0.01, cL2.z);

    // 6. Right Curb
    const cR1 = new THREE.Vector3().addVectors(p, normal.clone().multiplyScalar(halfW));
    const cR2 = new THREE.Vector3().addVectors(p, normal.clone().multiplyScalar(halfW + 0.6));
    cRVerts.push(cR1.x, p.y + 0.01, cR1.z);
    cRVerts.push(cR2.x, p.y + 0.01, cR2.z);

    // 7. Left Grass Ribbon (80m wide)
    const gL1 = new THREE.Vector3().addVectors(p, normal.clone().multiplyScalar(-halfW - 0.6 - grassW));
    const gL2 = new THREE.Vector3().addVectors(p, normal.clone().multiplyScalar(-halfW - 0.6));
    gLVerts.push(gL1.x, p.y - 0.005, gL1.z);
    gLVerts.push(gL2.x, p.y - 0.005, gL2.z);

    // 8. Right Grass Ribbon (80m wide)
    const gR1 = new THREE.Vector3().addVectors(p, normal.clone().multiplyScalar(halfW + 0.6));
    const gR2 = new THREE.Vector3().addVectors(p, normal.clone().multiplyScalar(halfW + 0.6 + grassW));
    gRVerts.push(gR1.x, p.y - 0.005, gR1.z);
    gRVerts.push(gR2.x, p.y - 0.005, gR2.z);

    if (i < numPoints - 1) {
      const idx = i * 2;
      aspIndices.push(idx, idx + 1, idx + 2, idx + 1, idx + 3, idx + 2);
      yelIndices.push(idx, idx + 1, idx + 2, idx + 1, idx + 3, idx + 2);
      wLIndices.push(idx, idx + 1, idx + 2, idx + 1, idx + 3, idx + 2);
      wRIndices.push(idx, idx + 1, idx + 2, idx + 1, idx + 3, idx + 2);
      cLIndices.push(idx, idx + 1, idx + 2, idx + 1, idx + 3, idx + 2);
      cRIndices.push(idx, idx + 1, idx + 2, idx + 1, idx + 3, idx + 2);
      gLIndices.push(idx, idx + 1, idx + 2, idx + 1, idx + 3, idx + 2);
      gRIndices.push(idx, idx + 1, idx + 2, idx + 1, idx + 3, idx + 2);
    }
  }

  const makeGeo = (v, idx, norm, uv) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(v, 3));
    if (norm) g.setAttribute('normal', new THREE.Float32BufferAttribute(norm, 3));
    if (uv) g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
  };

  return {
    asphaltGeo: makeGeo(aspVerts, aspIndices, aspNorms, aspUVs),
    yellowLineGeo: makeGeo(yelVerts, yelIndices),
    whiteLeftLineGeo: makeGeo(wLVerts, wLIndices),
    whiteRightLineGeo: makeGeo(wRVerts, wRIndices),
    curbLeftGeo: makeGeo(cLVerts, cLIndices),
    curbRightGeo: makeGeo(cRVerts, cRIndices),
    grassLeftGeo: makeGeo(gLVerts, gLIndices),
    grassRightGeo: makeGeo(gRVerts, gRIndices),
    centerPos
  };
}
