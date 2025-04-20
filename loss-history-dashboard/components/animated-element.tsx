"use client"

import { useState, useEffect, type ReactNode } from "react"
import { cn } from "@/lib/utils"

type AnimationType =
  | "fade-in"
  | "slide-up"
  | "slide-down"
  | "slide-left"
  | "slide-right"
  | "scale-in"
  | "rotate-in"
  | "bounce"
  | "pulse"
  | "shake"
  | "wiggle"
  | "float"

interface AnimatedElementProps {
  children: ReactNode
  type: AnimationType
  delay?: number
  duration?: number
  className?: string
  triggerOnce?: boolean
  threshold?: number
}

export function AnimatedElement({
  children,
  type,
  delay = 0,
  duration = 500,
  className,
  triggerOnce = true,
  threshold = 0.1,
}: AnimatedElementProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [hasAnimated, setHasAnimated] = useState(false)
  const [ref, setRef] = useState<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!ref) return
    if (triggerOnce && hasAnimated) return

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true)
          if (triggerOnce) {
            setHasAnimated(true)
            observer.disconnect()
          }
        } else if (!triggerOnce) {
          setIsVisible(false)
        }
      },
      { threshold },
    )

    observer.observe(ref)
    return () => observer.disconnect()
  }, [ref, triggerOnce, hasAnimated, threshold])

  const animationClasses = {
    "fade-in": "opacity-0 transition-opacity",
    "slide-up": "opacity-0 translate-y-8 transition-all",
    "slide-down": "opacity-0 -translate-y-8 transition-all",
    "slide-left": "opacity-0 translate-x-8 transition-all",
    "slide-right": "opacity-0 -translate-x-8 transition-all",
    "scale-in": "opacity-0 scale-95 transition-all",
    "rotate-in": "opacity-0 rotate-12 transition-all",
    bounce: "animate-bounce",
    pulse: "animate-pulse",
    shake: "animate-shake",
    wiggle: "animate-wiggle",
    float: "animate-float",
  }

  const visibleClasses = {
    "fade-in": "opacity-100",
    "slide-up": "opacity-100 translate-y-0",
    "slide-down": "opacity-100 translate-y-0",
    "slide-left": "opacity-100 translate-x-0",
    "slide-right": "opacity-100 translate-x-0",
    "scale-in": "opacity-100 scale-100",
    "rotate-in": "opacity-100 rotate-0",
    bounce: "",
    pulse: "",
    shake: "",
    wiggle: "",
    float: "",
  }

  return (
    <div
      ref={setRef}
      className={cn(animationClasses[type], isVisible && visibleClasses[type], className)}
      style={{
        transitionDelay: `${delay}ms`,
        transitionDuration: `${duration}ms`,
      }}
    >
      {children}
    </div>
  )
}

// Add keyframe animations to the global styles
const keyframeAnimations = `
@keyframes shake {
  0%, 100% { transform: translateX(0); }
  10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
  20%, 40%, 60%, 80% { transform: translateX(5px); }
}

@keyframes wiggle {
  0%, 100% { transform: rotate(-3deg); }
  50% { transform: rotate(3deg); }
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
}
`

// Add these animations to a style tag if they don't exist yet
if (typeof document !== "undefined") {
  const styleId = "animated-element-keyframes"
  if (!document.getElementById(styleId)) {
    const styleTag = document.createElement("style")
    styleTag.id = styleId
    styleTag.innerHTML = keyframeAnimations
    document.head.appendChild(styleTag)
  }
}

// Add Tailwind classes
const tailwindAnimations = {
  "animate-shake": "animation: shake 0.5s ease-in-out;",
  "animate-wiggle": "animation: wiggle 1s ease-in-out infinite;",
  "animate-float": "animation: float 3s ease-in-out infinite;",
}
