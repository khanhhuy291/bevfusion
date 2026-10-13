import test from 'node:test';
import assert from 'node:assert/strict';
import {transform,project,cameraSegments,bestCamera,orbitBasis,orbitProject,bevProject} from '../apps/vf_object_inspector/geometry.mjs';
const close=(a,b)=>a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-9,`${a} != ${b}`));
test('row-major transform applies rotation and translation',()=>close(transform([1,2,3],[[0,-1,0,10],[1,0,0,20],[0,0,1,30],[0,0,0,1]]),[8,21,33]));
test('camera intrinsic projection has expected pixels',()=>close(project([1,2,4],[[100,0,320],[0,100,240],[0,0,1]]),[345,290]));
test('near-plane crossing clips edges; empty selection is valid',()=>{
 const camera={width:640,height:480,intrinsic:[[100,0,320],[0,100,240],[0,0,1]],ego_to_camera:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]};
 const behind={corners:Array.from({length:8},()=>[1,1,-1])};assert.equal(cameraSegments(behind,camera).length,0);
 const crossing={corners:[[1,1,1],[1,-1,1],[1,-1,-1],[1,1,-1],[-1,1,1],[-1,-1,1],[-1,-1,-1],[-1,1,-1]]};
 const lines=cameraSegments(crossing,camera);assert.ok(lines.length>0);assert.ok(lines.flat(2).every(Number.isFinite));assert.equal(bestCamera(null,{camera}),null);
});
test('orbit keeps its target in the center and rejects points behind the eye',()=>{
 const basis=orbitBasis([3,4,1],10,.7,.4);const p=orbitProject([3,4,1],basis,800,600);close(p,[400,300,10]);
 const behind=basis.eye.map((v,i)=>v-basis.forward[i]);assert.equal(orbitProject(behind,basis,800,600),null);
});
test('BEV uses equal metric scale, forward up and left left',()=>{
 close(bevProject([0,0,0],[0,0,0],10,800,600),[400,300]);
 close(bevProject([1,0,0],[0,0,0],10,800,600),[400,270]);
 close(bevProject([0,1,0],[0,0,0],10,800,600),[370,300]);
});
