import { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';

interface HolographicCardProps extends React.HTMLAttributes<HTMLDivElement> {
  title: string;
  value: string | number;
  unit?: string;
  gradient?: string;
  glowColor?: string;
  badge?: {
    text: string;
    color: string;
  };
  onClick?: () => void;
}

export function HolographicCard({
  title,
  value,
  unit,
  gradient = 'from-blue-500/20 via-purple-500/20 to-pink-500/20',
  glowColor = 'rgba(59, 130, 246, 0.5)',
  badge,
  onClick,
  className,
  ...props
}: HolographicCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  const [glowPosition, setGlowPosition] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);
  const [scale, setScale] = useState(1);

  useEffect(() => {
    const card = cardRef.current;
    if (!card) return;

    const handleMouseMove = (e: MouseEvent) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      // Calculate rotation based on mouse position
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      const rotateX = (y - centerY) / 20;
      const rotateY = (centerX - x) / 20;
      
      setRotation({ x: rotateX, y: rotateY });
      setGlowPosition({ x, y });
    };

    const handleMouseLeave = () => {
      setRotation({ x: 0, y: 0 });
      setIsHovered(false);
      setScale(1);
    };

    const handleMouseEnter = () => {
      setIsHovered(true);
      setScale(1.02);
    };

    const handleClick = () => {
      setScale(0.98);
      setTimeout(() => setScale(1.02), 100);
      onClick?.();
    };

    card.addEventListener('mousemove', handleMouseMove);
    card.addEventListener('mouseleave', handleMouseLeave);
    card.addEventListener('mouseenter', handleMouseEnter);
    card.addEventListener('click', handleClick);

    return () => {
      card.removeEventListener('mousemove', handleMouseMove);
      card.removeEventListener('mouseleave', handleMouseLeave);
      card.removeEventListener('mouseenter', handleMouseEnter);
      card.removeEventListener('click', handleClick);
    };
  }, [onClick]);

  return (
    <div
      ref={cardRef}
      className={cn(
        'relative p-6 rounded-xl overflow-hidden transition-all duration-200',
        'bg-gradient-to-br backdrop-blur-sm',
        'border border-white/10',
        'transform-gpu cursor-pointer',
        'hover:shadow-2xl',
        gradient,
        className
      )}
      style={{
        transform: `perspective(1000px) rotateX(${rotation.x}deg) rotateY(${rotation.y}deg) scale(${scale})`,
      }}
      {...props}
    >
      {/* Holographic overlay */}
      <div
        className="absolute inset-0 opacity-50 pointer-events-none transition-opacity duration-300"
        style={{
          background: `radial-gradient(circle at ${glowPosition.x}px ${glowPosition.y}px, ${glowColor}, transparent 80%)`,
          opacity: isHovered ? 0.7 : 0.5,
        }}
      />

      {/* Scanning line effect */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'linear-gradient(transparent, rgba(255,255,255,0.1), transparent)',
          backgroundSize: '100% 200%',
          animation: 'scan 2s linear infinite',
        }}
      />

      {/* Content */}
      <div className="relative z-10 space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-white/80">{title}</h3>
          {badge && (
            <span
              className="px-2 py-1 text-xs font-medium rounded-full"
              style={{ backgroundColor: `${badge.color}20`, color: badge.color }}
            >
              {badge.text}
            </span>
          )}
        </div>
        <div className="flex items-baseline space-x-1">
          <span className="text-2xl font-bold text-white">{value}</span>
          {unit && <span className="text-sm text-white/60">{unit}</span>}
        </div>
      </div>

      {/* Holographic rings */}
      <div className="absolute inset-0 pointer-events-none">
        <div
          className="absolute inset-0 rounded-xl border border-white/20"
          style={{
            transform: `scale(${isHovered ? 1.05 : 1})`,
            transition: 'transform 0.3s ease-out',
          }}
        />
        <div
          className="absolute inset-0 rounded-xl border border-white/10"
          style={{
            transform: `scale(${isHovered ? 1.1 : 1.05})`,
            transition: 'transform 0.3s ease-out 0.1s',
          }}
        />
      </div>
    </div>
  );
} 