"use client";

/**
 * WebGL liquid-glass field — ported from the Stitch export.
 *
 * The fragment shader is kept as authored: a simplex-noise flow field warped by
 * an iterative refraction loop, with the noise treated as a height map so its
 * gradient can drive a specular highlight. That specular term is what separates
 * this from a blurred gradient — it produces moving glints that read as a
 * liquid surface catching light, which no amount of layered CSS blur achieves.
 *
 * Wrapped in React with the lifecycle the standalone page did not need: the
 * animation frame is cancelled, the ResizeObserver disconnected, the mouse
 * listener removed and the GL context explicitly released on unmount. Without
 * that, navigating away from the landing page would leave a full-screen render
 * loop running for the rest of the session.
 */

import { useEffect, useRef } from "react";

const VERT = `attribute vec2 a_position;
varying vec2 v_texCoord;
void main() {
  v_texCoord = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const FRAG = `precision highp float;
uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;

varying vec2 v_texCoord;

vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }
float snoise(vec2 v){
  const vec4 C = vec4(0.211324865405187, 0.366025403784439,
           -0.577350269189626, 0.024390243902439);
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);
  vec2 i1;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i, 289.0);
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
  + i.x + vec3(0.0, i1.x, 1.0 ));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
    dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 a0 = x - floor(x + 0.5);
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

void main() {
    vec2 uv = v_texCoord;
    vec2 p = uv * 2.0 - 1.0;
    p.x *= u_resolution.x / u_resolution.y;

    float t = u_time * 0.15;

    // Refractive distortion field.
    vec2 distort = uv;
    for(int i = 1; i < 4; i++) {
        float fi = float(i);
        distort.x += 0.3 / fi * sin(fi * 3.0 * uv.y + t + fi);
        distort.y += 0.3 / fi * cos(fi * 3.0 * uv.x + t + fi);
    }

    float n = snoise(distort * 2.5 + t * 0.5);
    n += snoise(distort * 5.0 - t * 0.3) * 0.5;

    // Treat the noise as a height map; its gradient gives a surface normal.
    float eps = 0.01;
    float n_x = snoise((distort + vec2(eps, 0.0)) * 2.5 + t * 0.5) - n;
    float n_y = snoise((distort + vec2(0.0, eps)) * 2.5 + t * 0.5) - n;
    vec2 grad = vec2(n_x, n_y) / eps;

    vec2 lightPos = (u_mouse.xy / u_resolution.xy) * 2.0 - 1.0;
    if(length(u_mouse) < 10.0) lightPos = vec2(sin(t), cos(t));

    float diff = max(dot(normalize(vec3(grad, 1.0)), normalize(vec3(lightPos, 1.0))), 0.0);
    float spec = pow(diff, 32.0);

    vec3 black = vec3(0.02, 0.02, 0.02);
    vec3 deepGrey = vec3(0.12, 0.12, 0.14);
    vec3 silkyWhite = vec3(0.95, 0.95, 0.98);

    vec3 color = mix(black, deepGrey, n * 0.5 + 0.5);

    float plumes = smoothstep(0.4, 0.9, n);
    color = mix(color, silkyWhite, plumes * 0.4);

    color += spec * 0.4;

    float vignette = 1.0 - length(p * 0.5);
    color *= smoothstep(0.0, 0.8, vignette);

    gl_FragColor = vec4(color, 1.0);
}`;

export function LiquidGlassShader() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl =
      (canvas.getContext("webgl") as WebGLRenderingContext | null) ??
      (canvas.getContext("experimental-webgl") as WebGLRenderingContext | null);
    if (!gl) return;

    const syncSize = () => {
      const w = canvas.clientWidth || 1280;
      const h = canvas.clientHeight || 720;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    };
    syncSize();

    const observer =
      typeof ResizeObserver !== "undefined" ? new ResizeObserver(syncSize) : null;
    observer?.observe(canvas);

    const compile = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      // A failed compile otherwise renders as a silent blank canvas, which is
      // indistinguishable from a layout problem and wastes a long time to find.
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.error("shader compile failed:", gl.getShaderInfoLog(s));
      }
      return s;
    };

    const prog = gl.createProgram()!;
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error("shader link failed:", gl.getProgramInfoLog(prog));
      return;
    }
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
      gl.STATIC_DRAW,
    );
    const pos = gl.getAttribLocation(prog, "a_position");
    gl.enableVertexAttribArray(pos);
    gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 0, 0);

    const uTime = gl.getUniformLocation(prog, "u_time");
    const uRes = gl.getUniformLocation(prog, "u_resolution");
    const uMouse = gl.getUniformLocation(prog, "u_mouse");

    const mouse = { x: canvas.width / 2, y: canvas.height / 2 };
    const onMouseMove = (event: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      if (rect.width && rect.height) {
        const nx = (event.clientX - rect.left) / rect.width;
        const ny = 1.0 - (event.clientY - rect.top) / rect.height;
        mouse.x = nx * canvas.width;
        mouse.y = ny * canvas.height;
      }
    };
    window.addEventListener("mousemove", onMouseMove);

    // Reduced motion freezes the field at a representative moment rather than
    // hiding it. The surface is the page's whole visual identity; removing it
    // would leave a blank black screen, while holding it still removes the
    // vestibular trigger and keeps the design intact.
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let raf = 0;
    const render = (t: number) => {
      if (!observer) syncSize();
      gl.viewport(0, 0, canvas.width, canvas.height);
      if (uTime) gl.uniform1f(uTime, reduced ? 12 : t * 0.001);
      if (uRes) gl.uniform2f(uRes, canvas.width, canvas.height);
      if (uMouse) gl.uniform2f(uMouse, mouse.x, mouse.y);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      if (!reduced) raf = requestAnimationFrame(render);
    };
    render(0);

    return () => {
      cancelAnimationFrame(raf);
      observer?.disconnect();
      window.removeEventListener("mousemove", onMouseMove);
      gl.deleteProgram(prog);
      gl.deleteBuffer(buf);
      // Deliberately not calling WEBGL_lose_context here. loseContext() kills
      // the context permanently for this canvas, and getContext() afterwards
      // hands back the same dead one — so under StrictMode's mount/unmount/
      // mount in development the second mount renders nothing at all. Dropping
      // the canvas from the DOM releases the context on its own.
    };
  }, []);

  return (
    <div className="fixed inset-0 -z-10 h-full w-full" aria-hidden>
      <canvas ref={canvasRef} className="block h-full w-full" />
    </div>
  );
}
