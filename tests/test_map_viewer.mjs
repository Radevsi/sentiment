// Run with: node --test tests/test_map_viewer.mjs
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../src/finance_sentiment/map_static/viewer.js',import.meta.url),'utf8');
const {screenPoint,worldPoint,pickPoints,zoomAt}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
test('projection-to-screen inversion respects pan, zoom and aspect ratio',()=>{
 const camera={x:10,y:-4,scale:20};
 const screen=screenPoint(12,-1,camera,1200,500);
 assert.deepEqual(screen,[640,190]);
 assert.deepEqual(worldPoint(...screen,camera,1200,500),[12,-1]);
});
test('click preserves exact row ID and exposes overlapping points',()=>{
 const points=new Float32Array([8,8,0,0,0,0,-8,-8]);
 assert.deepEqual(pickPoints(points,100,50,{x:0,y:0,scale:10},200,100),{total:2,ids:[1,2]});
 assert.deepEqual(pickPoints(points,180,-30,{x:0,y:0,scale:10},200,100).ids,[0]);
});
test('all 219584 points participate in selection, including the final row',()=>{
 const points=new Float32Array(219584*2);points[points.length-2]=123;points[points.length-1]=100;
 assert.deepEqual(pickPoints(points,223,-50,{x:0,y:0,scale:1},200,100).ids,[219583]);
});
test('wheel zoom preserves world location under cursor',()=>{
 const before={x:12,y:20,scale:5},after=zoomAt(before,180,75,2,800,600);
 assert.deepEqual(worldPoint(180,75,before,800,600),worldPoint(180,75,after,800,600));
});
