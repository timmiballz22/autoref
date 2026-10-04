import type {ReactNode} from 'react';
import {interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {colors, font} from './theme';

export const Grain: React.FC = () => (
  <svg style={{position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: 0.1, mixBlendMode: 'multiply'}}>
    <filter id="noise"><feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="3" seed="9"/></filter>
    <rect width="100%" height="100%" filter="url(#noise)"/>
  </svg>
);

export const NumberTag: React.FC<{children: ReactNode}> = ({children}) => (
  <span style={{font: `700 23px ${font}`, letterSpacing: 2, border: `2px solid ${colors.ink}`, borderRadius: 40, padding: '10px 19px'}}>{children}</span>
);

export const RevealText: React.FC<{children: ReactNode; delay?: number; style?: React.CSSProperties}> = ({children, delay = 0, style}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const p = spring({frame: frame - delay, fps, config: {damping: 18, stiffness: 110}});
  return <div style={{opacity: p, transform: `translateY(${(1 - p) * 42}px)`, ...style}}>{children}</div>;
};

export const Progress: React.FC<{index: number; total?: number}> = ({index, total = 7}) => (
  <div style={{position: 'absolute', bottom: 56, left: 70, right: 70, display: 'flex', alignItems: 'center', gap: 22, font: `700 17px ${font}`}}>
    <span>0{index}</span>
    <div style={{height: 3, background: `${colors.ink}25`, flex: 1}}><div style={{height: '100%', background: colors.ink, width: `${index / total * 100}%`}}/></div>
    <span>0{total}</span>
  </div>
);

export const Orbit: React.FC<{radius: number; speed?: number; color?: string}> = ({radius, speed = 1, color = colors.coral}) => {
  const frame = useCurrentFrame();
  const angle = frame * speed * 0.025;
  return <div style={{position: 'absolute', width: radius * 2, height: radius * 2, border: `2px solid ${colors.ink}22`, borderRadius: '50%'}}>
    <div style={{position: 'absolute', left: radius - 9 + Math.cos(angle) * radius, top: radius - 9 + Math.sin(angle) * radius, width: 18, height: 18, borderRadius: '50%', background: color}}/>
  </div>;
};

export const TimelineLine: React.FC<{progress: number}> = ({progress}) => (
  <div style={{position: 'absolute', left: 160, right: 160, top: 545, height: 4, background: colors.ink}}>
    <div style={{position: 'absolute', right: -2, top: -7, borderLeft: `16px solid ${colors.ink}`, borderTop: '9px solid transparent', borderBottom: '9px solid transparent'}}/>
    <div style={{height: 16, width: 16, background: colors.coral, borderRadius: 16, transform: `translate(${progress * 1570}px, -6px)`, boxShadow: `0 0 0 8px ${colors.paper}`}}/>
  </div>
);

export const wipe = (frame: number, duration: number) => interpolate(frame, [duration - 18, duration], [0, 100], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
