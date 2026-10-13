export const EDGES = [[0,4],[1,5],[2,6],[3,7],[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4]];
export const FOOTPRINT = [2,3,7,6];
export function transform(point,m) { return [0,1,2].map(i=>m[i][0]*point[0]+m[i][1]*point[1]+m[i][2]*point[2]+m[i][3]); }
export function project(point,k) { const q=k.map(row=>row.reduce((sum,v,i)=>sum+v*point[i],0)); return [q[0]/q[2],q[1]/q[2]]; }
export function cameraSegments(record,camera) {
  const pts=record.corners.map(p=>transform(p,camera.ego_to_camera));
  return EDGES.flatMap(([i,j])=>{
    let a=pts[i],b=pts[j]; const near=.15;
    if(a[2]<near&&b[2]<near)return [];
    if(a[2]<near){const t=(near-a[2])/(b[2]-a[2]); a=a.map((v,n)=>v+t*(b[n]-v));}
    if(b[2]<near){const t=(near-b[2])/(a[2]-b[2]); b=b.map((v,n)=>v+t*(a[n]-v));}
    return [[project(a,camera.intrinsic),project(b,camera.intrinsic)]];
  });
}
export function projectedBounds(record,camera) {
  const pts=cameraSegments(record,camera).flat(); if(!pts.length)return null;
  const low=[Math.max(0,Math.min(...pts.map(p=>p[0]))),Math.max(0,Math.min(...pts.map(p=>p[1])))];
  const high=[Math.min(camera.width,Math.max(...pts.map(p=>p[0]))),Math.min(camera.height,Math.max(...pts.map(p=>p[1])))];
  return high[0]>low[0]&&high[1]>low[1] ? [...low,...high] : null;
}
export function bestCamera(record,cameras) {
  if(!record)return null;
  let best=null,area=0;for(const [name,camera] of Object.entries(cameras)){
    const b=projectedBounds(record,camera);const a=b?(b[2]-b[0])*(b[3]-b[1]):0;
    if(a>area){area=a;best=name;}
  }return best;
}
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const norm=a=>{const d=Math.hypot(...a);return a.map(v=>v/d);};
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export function orbitBasis(target,distance,yaw,pitch) {
  const eye=target.map((v,i)=>v+distance*[Math.cos(yaw)*Math.cos(pitch),Math.sin(yaw)*Math.cos(pitch),Math.sin(pitch)][i]);
  const forward=norm(target.map((v,i)=>v-eye[i]));const right=norm(cross(forward,[0,0,1]));const up=cross(right,forward);
  return {eye,forward,right,up};
}
export function orbitProject(point,basis,w,h) {
  const p=point.map((v,i)=>v-basis.eye[i]),z=dot(p,basis.forward);if(z<.1)return null;
  const f=Math.min(w,h)*1.1;return [w/2+dot(p,basis.right)*f/z,h/2-dot(p,basis.up)*f/z,z];
}
export function bevProject(point,center,radius,w,h) {
  const scale=Math.min(w,h)/(2*radius);return [w/2-(point[1]-center[1])*scale,h/2-(point[0]-center[0])*scale];
}
