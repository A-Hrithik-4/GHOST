import * as THREE from 'three';

/**
 * trajectoryUtils.js
 * 
 * Isolated conversion & interpolation layer for GHOST 3D visualization.
 * 
 * Coordinate Transformation:
 * GHOST local metric system (Equirectangular relative to P0):
 *   x_metric = Easting distance (m)
 *   y_metric = Northing distance (m)
 * 
 * Three.js 3D Coordinate System:
 *   X_3d = x_metric (East / West)
 *   Y_3d = 0.45m (Height above road surface)
 *   Z_3d = y_metric (North / South)
 * 
 * Heading Conversion:
 *   Compass heading angle theta_deg (0° = North, 90° = East)
 *   Three.js yaw angle: rotation_Y = -radians(theta_deg)
 */

export function metricToThreePosition(xMetric, yMetric, height = 0.45) {
  return [xMetric, height, yMetric];
}

export function headingToThreeRotation(headingDeg) {
  const rad = (headingDeg * Math.PI) / 180.0;
  return [0, -rad, 0];
}

/**
 * Linear Interpolation between trajectory data samples
 */
export function getInterpolatedSample(samples, currentTime) {
  if (!samples || samples.length === 0) return null;

  if (currentTime <= samples[0].t) {
    return samples[0];
  }

  const lastSample = samples[samples.length - 1];
  if (currentTime >= lastSample.t) {
    return lastSample;
  }

  // Binary search for surrounding timesteps
  let low = 0;
  let high = samples.length - 1;

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    if (samples[mid].t === currentTime) {
      return samples[mid];
    } else if (samples[mid].t < currentTime) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  const idx1 = Math.max(0, high);
  const idx2 = Math.min(samples.length - 1, low);

  const s1 = samples[idx1];
  const s2 = samples[idx2];

  if (s1.t === s2.t) return s1;

  const alpha = (currentTime - s1.t) / (s2.t - s1.t);

  // Helper for angular interpolation (handling degree wrap around)
  const interpAngle = (a1, a2) => {
    let diff = (a2 - a1 + 180) % 360 - 180;
    return a1 + diff * alpha;
  };

  return {
    t: currentTime,
    state: currentTime >= 30.0 && currentTime < 90.0 ? 'OUTAGE' : (currentTime >= 90.0 ? 'RESTORED' : 'AVAILABLE'),
    gt: {
      x: THREE.MathUtils.lerp(s1.gt.x, s2.gt.x, alpha),
      z: THREE.MathUtils.lerp(s1.gt.z, s2.gt.z, alpha),
      h: interpAngle(s1.gt.h, s2.gt.h)
    },
    p4: {
      x: THREE.MathUtils.lerp(s1.p4.x, s2.p4.x, alpha),
      z: THREE.MathUtils.lerp(s1.p4.z, s2.p4.z, alpha),
      h: interpAngle(s1.p4.h, s2.p4.h),
      err: THREE.MathUtils.lerp(s1.p4.err, s2.p4.err, alpha)
    },
    p6: {
      x: THREE.MathUtils.lerp(s1.p6.x, s2.p6.x, alpha),
      z: THREE.MathUtils.lerp(s1.p6.z, s2.p6.z, alpha),
      h: interpAngle(s1.p6.h, s2.p6.h),
      err: THREE.MathUtils.lerp(s1.p6.err, s2.p6.err, alpha),
      spd: THREE.MathUtils.lerp(s1.p6.spd, s2.p6.spd, alpha),
      bias: THREE.MathUtils.lerp(s1.p6.bias, s2.p6.bias, alpha)
    }
  };
}
