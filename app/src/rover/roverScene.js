// Three.js view of a mobile-robot map: floor, walls, furniture, unknown
// obstacles, named places, the rover (+ measured "ghost" in live mode),
// lidar rays and the planned path. Shares the arm view's WebGL renderer.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const W2T = (x, y, z = 0) => new THREE.Vector3(x, z, -y);

function label(text) {
  const c = document.createElement("canvas"); c.width = 256; c.height = 64;
  const g = c.getContext("2d");
  g.font = "bold 30px system-ui, sans-serif"; g.fillStyle = "rgba(0,0,0,.55)";
  const w = g.measureText(text).width + 24; g.fillRect((256 - w) / 2, 8, w, 48);
  g.fillStyle = "#e0f7fa"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(text, 128, 33);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  s.scale.set(0.9, 0.225, 1);
  return s;
}

function roverMesh(color, opacity = 1) {
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color, transparent: opacity < 1, opacity, roughness: 0.6 });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.105, 0.105, 0.07, 32), mat);
  body.position.y = 0.06; g.add(body);
  const head = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.04, 16), new THREE.MeshStandardMaterial({ color: 0x111111, transparent: opacity < 1, opacity }));
  head.position.y = 0.15; g.add(head);
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.03, 0.07, 12), new THREE.MeshStandardMaterial({ color: 0x00e5ff, transparent: opacity < 1, opacity }));
  nose.rotation.z = -Math.PI / 2; nose.position.set(0.09, 0.1, 0); g.add(nose);
  for (const s of [1, -1]) {
    const w = new THREE.Mesh(new THREE.CylinderGeometry(0.033, 0.033, 0.018, 20), new THREE.MeshStandardMaterial({ color: 0x222222, transparent: opacity < 1, opacity }));
    w.rotation.x = Math.PI / 2; w.position.set(0, 0.033, -s * 0.08); g.add(w);
  }
  return g;
}

export class RoverScene {
  constructor(renderer, canvas) {
    this.renderer = renderer;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 50);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enabled = false;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.5));
    const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(3, 8, 2); this.scene.add(sun);
    this.world = new THREE.Group(); this.scene.add(this.world);
    this.unknownGroup = new THREE.Group(); this.scene.add(this.unknownGroup);
    this.rover = roverMesh(0x37474f); this.scene.add(this.rover);
    const halo = new THREE.Mesh(new THREE.RingGeometry(0.15, 0.2, 32), new THREE.MeshBasicMaterial({ color: 0xffab00, side: THREE.DoubleSide }));
    halo.rotation.x = -Math.PI / 2; halo.position.y = 0.004; this.rover.add(halo);
    this.ghost = roverMesh(0x00e5ff, 0.35); this.ghost.visible = false; this.scene.add(this.ghost);
    this.rays = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xff5252, transparent: true, opacity: 0.55 }));
    this.scene.add(this.rays);
    this.pathLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x00e5ff }));
    this.scene.add(this.pathLine);
    this.goalRing = new THREE.Mesh(new THREE.RingGeometry(0.14, 0.2, 32), new THREE.MeshBasicMaterial({ color: 0x76ff03, side: THREE.DoubleSide }));
    this.goalRing.rotation.x = -Math.PI / 2; this.goalRing.visible = false; this.scene.add(this.goalRing);
  }
  setMap(map) {
    this.map = map;
    this.world.clear();
    this.scene.background = new THREE.Color(map.background);
    const [w, h] = map.size;
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshStandardMaterial({ color: map.floor_color, roughness: 0.95 }));
    floor.rotation.x = -Math.PI / 2; floor.position.copy(W2T(w / 2, h / 2)); this.world.add(floor);
    const grid = new THREE.GridHelper(Math.max(w, h), Math.round(Math.max(w, h) * 2), 0x777777, 0x555555);
    grid.position.copy(W2T(w / 2, h / 2, 0.001)); grid.material.transparent = true; grid.material.opacity = 0.25; this.world.add(grid);
    const box = (b, height, color) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(b.size[0], height, b.size[1]), new THREE.MeshStandardMaterial({ color, roughness: 0.85 }));
      m.position.copy(W2T(b.pos[0], b.pos[1], height / 2)); return m;
    };
    for (const b of map.walls) this.world.add(box(b, 0.35, map.wall_color));
    for (const o of map.obstacles) this.world.add(box(o, o.height, o.color));
    for (const p of map.places) {
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.1, 0.13, 24), new THREE.MeshBasicMaterial({ color: 0x4dd0e1, side: THREE.DoubleSide, transparent: true, opacity: 0.6 }));
      ring.rotation.x = -Math.PI / 2; ring.position.copy(W2T(p.pos[0], p.pos[1], 0.003)); this.world.add(ring);
      const l = label(p.names_id[0]); l.position.copy(W2T(p.pos[0], p.pos[1], 0.45)); this.world.add(l);
    }
    this.camera.position.copy(W2T(w / 2, -h * 0.35, Math.max(w, h) * 1.05));
    this.controls.target.copy(W2T(w / 2, h / 2, 0));
    this.controls.update();
  }
  setUnknown(list) {
    this.unknownGroup.clear();
    for (const o of list) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(o.size[0], o.height, o.size[1]), new THREE.MeshStandardMaterial({ color: o.color, roughness: 0.7 }));
      m.position.copy(W2T(o.pos[0], o.pos[1], o.height / 2)); this.unknownGroup.add(m);
    }
  }
  resize(w, h) { this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); }
  render({ pose, ghost = null, scan = null, path = null, goal = null }) {
    this.rover.position.copy(W2T(pose[0], pose[1])); this.rover.rotation.y = pose[2];
    this.ghost.visible = !!ghost;
    if (ghost) { this.ghost.position.copy(W2T(ghost[0], ghost[1])); this.ghost.rotation.y = ghost[2]; }
    if (scan) {
      const pts = [];
      scan.forEach((r, k) => {
        const a = pose[2] + (k * 2 * Math.PI) / scan.length;
        pts.push(W2T(pose[0], pose[1], 0.15), W2T(pose[0] + r * Math.cos(a), pose[1] + r * Math.sin(a), 0.15));
      });
      this.rays.geometry.setFromPoints(pts);
    }
    this.pathLine.visible = !!path;
    if (path) this.pathLine.geometry.setFromPoints(path.map((p) => W2T(p[0], p[1], 0.02)));
    this.goalRing.visible = !!goal;
    if (goal) this.goalRing.position.copy(W2T(goal[0], goal[1], 0.004));
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

/** Polar lidar plot for the HUD thumbnail. */
export function drawScan(canvas, scan, rmax = 3.5, shieldFront = null) {
  const g = canvas.getContext("2d"), W = canvas.width, H = canvas.height, cx = W / 2, cy = H / 2, s = Math.min(W, H) / 2 / rmax;
  g.fillStyle = "#0b1016"; g.fillRect(0, 0, W, H);
  g.strokeStyle = "#263238"; for (const r of [1, 2, 3]) { g.beginPath(); g.arc(cx, cy, r * s, 0, 2 * Math.PI); g.stroke(); }
  g.fillStyle = "#ff5252";
  scan.forEach((r, k) => {
    const a = (k * 2 * Math.PI) / scan.length; // robot frame: forward = up
    const x = cx - Math.sin(a) * r * s, y = cy - Math.cos(a) * r * s;
    g.fillRect(x - 1.5, y - 1.5, 3, 3);
  });
  g.fillStyle = "#00e5ff"; g.beginPath(); g.moveTo(cx, cy - 7); g.lineTo(cx - 5, cy + 5); g.lineTo(cx + 5, cy + 5); g.fill();
  if (shieldFront !== null) { g.fillStyle = "#b0bec5"; g.font = "10px system-ui"; g.fillText(`depan ${shieldFront.toFixed(2)} m`, 4, H - 4); }
}
