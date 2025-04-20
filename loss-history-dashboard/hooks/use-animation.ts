"use client"

import { useState, useEffect, useCallback } from "react"

type AnimationOptions = {
  duration?: number
  delay?: number
  easing?: string
  onComplete?: () => void
}

export function useAnimation(initialState = false, options: AnimationOptions = {}) {
  const [isAnimating, setIsAnimating] = useState(initialState)
  const [hasAnimated, setHasAnimated] = useState(initialState)

  const { duration = 300, delay = 0, easing = "ease", onComplete } = options

  const startAnimation = useCallback(() => {
    setIsAnimating(true)

    const timer = setTimeout(() => {
      setHasAnimated(true)
      setIsAnimating(false)
      if (onComplete) onComplete()
    }, duration + delay)

    return () => clearTimeout(timer)
  }, [duration, delay, onComplete])

  const resetAnimation = useCallback(() => {
    setIsAnimating(false)
    setHasAnimated(false)
  }, [])

  return {
    isAnimating,
    hasAnimated,
    startAnimation,
    resetAnimation,
    style: {
      transition: `all ${duration}ms ${delay}ms ${easing}`,
    },
  }
}

export function useEntranceAnimation(options: AnimationOptions = {}, triggerOnMount = true) {
  const animation = useAnimation(false, options)

  useEffect(() => {
    if (triggerOnMount) {
      animation.startAnimation()
    }
  }, [triggerOnMount, animation])

  return animation
}

export function useHoverAnimation(options: AnimationOptions = {}) {
  const [isHovered, setIsHovered] = useState(false)
  const animation = useAnimation(false, options)

  useEffect(() => {
    if (isHovered) {
      animation.startAnimation()
    } else {
      animation.resetAnimation()
    }
  }, [isHovered, animation])

  return {
    ...animation,
    handleMouseEnter: () => setIsHovered(true),
    handleMouseLeave: () => setIsHovered(false),
  }
}
