"use client";
import React, { useEffect, useRef, useState } from "react";
import p5Types from "p5";

type P5jsContainerRef = HTMLDivElement;
type P5jsSketch = (p: p5Types, parentRef: P5jsContainerRef) => void;
type P5jsContainer = ({ sketch }: { sketch: P5jsSketch }) => React.JSX.Element;

export const P5Container: P5jsContainer = ({ sketch }) => {
  const parentRef = useRef<P5jsContainerRef>();

  const [isMounted, setIsMounted] = useState<boolean>(false);

  // When the component mounts, set isMounted to true
  useEffect(() => {
    setIsMounted(true);
    parentRef.current = document.createElement("div");
  }, []);

  // Create the p5 instance when the component mounts
  useEffect(() => {
    if (!isMounted) return;

    const initP5 = async () => {
      // Importing p5 here to prevent server-side rendering
      const p5 = (await import("p5")).default;

      // Create the p5 instance
      return new p5(sketch, parentRef.current);
    };
  }, [isMounted, sketch]);

  return <div ref={parentRef}></div>;
};

export const mySketch: P5jsSketch = (p5, parentRef) => {
  let parentStyle: CSSStyleDeclaration;
  let canvasHeight: number;
  let canvasWidth: number;

  p5.setup = () => {
    // get and set the canvas size inside the parent
    // let parentStyle = window.getComputedStyle(parentRef.current);
    // canvasWidth = parseInt(parentStyle.width) * 0.99;
    // canvasHeight = parseInt(parentStyle.width) * 0.4;
    // p5.createCanvas(canvasWidth, canvasHeight).parent(parentRef);
  };

  p5.draw = () => {
    p5.background(220);
    p5.fill(255, 0, 0);
    p5.ellipse(p5.width / 2, p5.height / 2, 50, 50);
  };
};
