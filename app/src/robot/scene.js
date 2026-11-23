// Three.js scene: table, arm (rendered from kinematics), cubes, pads,
// a "ghost" arm for the measured state of a live robot, and an overhead
// orthographic camera used by the vision module.
// World frame (robot): x forward, y left, z up  ->  three: (x, z, -y).
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { jointPositions } from "./kinematics.js";

export const W2T = (p) => new THREE.Vector3(p[0], p[2], -p[1]);
export const CUBE = 0.03;
export const COLOR_HEX = { red: 0xe53935, green: 0x43a047, blue: 0x1e88e5, yellow: 0xfdd835, purple: 0x8e24aa,
  orange: 0xfa7317, cyan: 0x00bfa5 };

export const VISION_BOUNDS = { xMin: -0.02, xMax: 0.38, yMin: -0.25, yMax: 0.25 };

function link(material, radius) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 1, 16), material);
  m.castShadow = true;
  return m;
}
function placeLink(mesh, a, b) {
  const A = W2T(a), B = W2T(b);
  const d = new THREE.Vector3().subVectors(B, A);
  const len = Math.max(d.length(), 1e-6);
  mesh.position.copy(A).addScaledVector(d, 0.5);
  mesh.scale.set(1, len, 1);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
}

class ArmView {
  constructor(scene, { color = 0xb0bec5, opacity = 1 } = {}) {
    const mat = new THREE.MeshStandardMaterial({ color, metalness: 0.3, roughness: 0.5, transparent: opacity < 1, opacity });
    const accent = new THREE.MeshStandardMaterial({ color: 0x37474f, transparent: opacity < 1, opacity });
    this.group = new THREE.Group();
    this.base = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.055, 0.03, 24), accent);
    this.base.position.y = 0.015;
    this.links = [link(accent, 0.022), link(mat, 0.018), link(mat, 0.015), link(mat, 0.012)];
    this.joints = [0, 1, 2, 3].map(() => new THREE.Mesh(new THREE.SphereGeometry(0.02, 16, 12), accent));
    this.fingers = [0, 1].map(() => new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.035, 0.012), accent));
    this.group.add(this.base, ...this.links, ...this.joints, ...this.fingers);
    scene.add(this.group);
  }
  update(q) {
    const p = jointPositions(q);
    for (let i = 0; i < 4; i++) placeLink(this.links[i], p[i], p[i + 1]);
    for (let i = 0; i < 4; i++) this.joints[i].position.copy(W2T(p[i + 1]));
    // fingers: open width from gripper q[5] in [0,1], spread perpendicular to the pan direction
    const tip = p[4], wrist = p[3];
    const dir = W2T(tip).sub(W2T(wrist)).normalize();
    const side = new THREE.Vector3(-Math.sin(q[0]), 0, -Math.cos(q[0])).applyAxisAngle(dir, q[4]);
    const half = 0.006 + 0.016 * q[5];
    this.fingers.forEach((f, k) => {
      f.position.copy(W2T(tip)).addScaledVector(side, k ? half : -half).addScaledVector(dir, -0.012);
      f.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    });
  }
  set visible(v) { this.group.visible = v; }
}

export class RobotScene {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0f141a);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10);
    this.camera.position.set(0.55, 0.42, 0.45);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0.16, 0.05, 0);
    this.controls.update();

    this.hemi = new THREE.HemisphereLight(0xffffff, 0x334455, 1.6);
    this.scene.add(this.hemi);
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    this.sun = sun;
    sun.position.set(0.3, 1.2, 0.4);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    Object.assign(sun.shadow.camera, { left: -0.5, right: 0.5, top: 0.5, bottom: -0.5 });
    this.scene.add(sun);

    const table = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.02, 0.8),
      new THREE.MeshStandardMaterial({ color: 0x4e4a46, roughness: 0.9 })); // low-saturation: never confused with a colored object
    table.position.set(0.15, -0.01, 0);
    table.receiveShadow = true;
    this.scene.add(table);
    this.table = table;
    const grid = new THREE.GridHelper(0.8, 16, 0x6b6661, 0x5a5652);
    grid.position.set(0.15, 0.0005, 0);
    this.scene.add(grid);
    this.grid = grid;
    this.obstacleGroup = new THREE.Group();
    this.scene.add(this.obstacleGroup);
    this.env = null;

    this.arm = new ArmView(this.scene);
    this.ghost = new ArmView(this.scene, { color: 0x00e5ff, opacity: 0.35 });
    this.ghost.visible = false;
    this.targetMarker = new THREE.Mesh(new THREE.SphereGeometry(0.006, 12, 8),
      new THREE.MeshBasicMaterial({ color: 0xffffff }));
    this.scene.add(this.targetMarker);

    this.objects = {}; // name -> {mesh, kind, pos:[x,y,z], color}
    // overhead camera for perception (orthographic, looks straight down)
    const b = VISION_BOUNDS;
    this.visionCam = new THREE.OrthographicCamera(-(b.yMax - b.yMin) / 2, (b.yMax - b.yMin) / 2,
      (b.xMax - b.xMin) / 2, -(b.xMax - b.xMin) / 2, 0.01, 2);
    this.visionCam.position.set((b.xMin + b.xMax) / 2, 1.0, 0);
    this.visionCam.up.set(1, 0, 0); // image "up" = robot +x (forward)
    this.visionCam.lookAt((b.xMin + b.xMax) / 2, 0, 0);
    this.visionW = 160; this.visionH = 128;
    this.visionTarget = new THREE.WebGLRenderTarget(this.visionW, this.visionH);
    // encode like a real camera (sRGB); a linear target shifts hues (yellow read as orange)
    this.visionTarget.texture.colorSpace = THREE.SRGBColorSpace;
  }

  /** Render an environment preset (app/public/envs/*.json), optionally with
   *  visual domain randomization {light, tableTint, camShift, camRot}. */
  applyEnv(env, vdr = null) {
    this.env = env;
    const light = vdr ? vdr.light : 1;
    this.scene.background = new THREE.Color(env.background);
    const [w, d] = env.table.size, [cx, cy] = env.table.center;
    this.table.scale.set(w / 0.8, 1, d / 0.8);
    this.table.position.set(cx, -0.01, -cy);
    const col = new THREE.Color(env.table.color);
    if (vdr) { const hsl = {}; col.getHSL(hsl); col.setHSL(hsl.h, hsl.s, Math.min(0.9, Math.max(0.05, hsl.l + vdr.tableTint))); }
    this.table.material.color.copy(col);
    this.grid.scale.set(w / 0.8, 1, d / 0.8);
    this.grid.position.set(cx, 0.0005, -cy);
    const tint = new THREE.Color(env.lighting.tint || "#ffffff");
    this.hemi.intensity = env.lighting.ambient * light;
    this.hemi.color.copy(tint);
    this.sun.intensity = env.lighting.sun * light;
    this.sun.color.copy(tint);
    this.sun.position.copy(W2T(env.lighting.sun_pos));
    this.obstacleGroup.clear();
    for (const o of env.obstacles) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(o.size[0], o.size[2], o.size[1]),
        new THREE.MeshStandardMaterial({ color: o.color, roughness: 0.85 }));
      m.position.copy(W2T(o.pos));
      m.castShadow = !o.visual_only; m.receiveShadow = true;
      this.obstacleGroup.add(m);
    }
    // camera mis-calibration (the detector still assumes the nominal mapping)
    const b = VISION_BOUNDS, sh = vdr ? vdr.camShift : [0, 0], rot = vdr ? vdr.camRot : 0;
    this.visionCam.position.set((b.xMin + b.xMax) / 2 + sh[0], 1.0, -sh[1]);
    this.visionCam.up.set(Math.cos(rot), 0, Math.sin(rot));
    this.visionCam.lookAt((b.xMin + b.xMax) / 2 + sh[0], 0, -sh[1]);
  }

  resize(w, h) {
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  clearObjects() {
    for (const o of Object.values(this.objects)) this.scene.remove(o.mesh);
    this.objects = {};
  }
  addCube(color, pos, yaw = Math.random(), shape = "cube") {
    const geo = shape === "cylinder" ? new THREE.CylinderGeometry(CUBE / 2, CUBE / 2, CUBE, 24) : new THREE.BoxGeometry(CUBE, CUBE, CUBE);
    const mesh = new THREE.Mesh(geo,
      new THREE.MeshLambertMaterial({ color: COLOR_HEX[color] }));
    mesh.castShadow = true;
    this.scene.add(mesh);
    this.objects[color] = { mesh, kind: "cube", shape, color, pos: [pos[0], pos[1], pos.length > 2 ? pos[2] : CUBE / 2], yaw };
  }
  addPad(color, pos) {
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.002, 32),
      new THREE.MeshLambertMaterial({ color: COLOR_HEX[color] }));
    mesh.receiveShadow = true;
    this.scene.add(mesh);
    this.objects[color] = { mesh, kind: "pad", color, pos: [pos[0], pos[1], 0.001] };
  }
  syncObjects() {
    for (const o of Object.values(this.objects)) {
      o.mesh.position.copy(W2T(o.pos));
      if (o.kind === "cube") o.mesh.rotation.y = o.yaw || 0;
    }
  }

  /** Render the overhead camera and return RGBA pixels (Uint8Array, row 0 = bottom). */
  captureVision() {
    this.syncObjects(); // object meshes must reflect current world state before the shot
    const prevGhost = this.ghost.group.visible, prevMarker = this.targetMarker.visible;
    this.ghost.group.visible = false; this.targetMarker.visible = false;
    this.renderer.setRenderTarget(this.visionTarget);
    this.renderer.render(this.scene, this.visionCam);
    const px = new Uint8Array(this.visionW * this.visionH * 4);
    this.renderer.readRenderTargetPixels(this.visionTarget, 0, 0, this.visionW, this.visionH, px);
    this.renderer.setRenderTarget(null);
    this.ghost.group.visible = prevGhost; this.targetMarker.visible = prevMarker;
    return px;
  }

  render(q, { ghostQ = null, target = null } = {}) {
    this.arm.update(q);
    if (ghostQ) { this.ghost.visible = true; this.ghost.update(ghostQ); } else this.ghost.visible = false;
    this.targetMarker.visible = !!target;
    if (target) this.targetMarker.position.copy(W2T(target));
    this.syncObjects();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
