import React, { useMemo } from 'react';
import { Line } from '@react-three/drei';

/**
 * TrajectoryLine.jsx
 * 
 * Renders thick, high-contrast 3D trajectory polylines using @react-three/drei Line.
 */

export default function TrajectoryLine({ points = [], color = '#2563EB', linewidth = 4, opacity = 1.0 }) {
  const formattedPoints = useMemo(() => {
    if (!points || points.length < 2) return [];
    return points.map((p) => [p[0], p[1] || 0.08, p[2]]);
  }, [points]);

  if (!formattedPoints || formattedPoints.length < 2) return null;

  return (
    <Line
      points={formattedPoints}
      color={color}
      lineWidth={linewidth}
      transparent={opacity < 1.0}
      opacity={opacity}
    />
  );
}
