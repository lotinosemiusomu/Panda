from scene import Scene
import taichi as ti
from taichi.math import *
scene = Scene(voxel_edges=0.01, exposure=1)
scene.set_floor(-64, (1.0, 1.0, 1.0))
scene.set_background_color((0, 0, 0))
scene.set_directional_light((-1, 1, 0.3), 0.0, (1, 1, 1))
@ti.func
def rgb(r,g,b):
    return vec3(r/255.0, g/255.0, b/255.0)
@ti.func
def proj_plane(o, n, t, p):
    y = dot(p-o,n);xz=p-(o+n*y);bt=cross(t,n);return vec3(dot(xz,t), y, dot(xz, bt))
@ti.func
def elli(rx,ry,rz,p1_unused,p2_unused,p3_unused,p):
    r = p/vec3(rx,ry,rz); return ti.sqrt(dot(r,r))<1
@ti.func
def cyli(r1,h,r2,round, cone, hole_unused, p):
    ms=min(r1,min(h,r2));rr=ms*round;rt=mix(cone*(max(ms-rr,0)),0,float(h-p.y)*0.5/h);r=vec2(p.x/r1,p.z/r2)
    d=vec2((r.norm()-1.0)*ms+rt,ti.abs(p.y)-h)+rr; return min(max(d.x,d.y),0.0)+max(d,0.0).norm()-rr<0
@ti.func
def box(x, y, z, round, cone, unused, p):
    ms=min(x,min(y,z));rr=ms*round;rt=mix(cone*(max(ms-rr,0)),0,float(y-p.y)*0.5/y);q=ti.abs(p)-vec3(x-rt,y,z-rt)+rr
    return ti.max(q, 0.0).norm() + ti.min(ti.max(q.x, ti.max(q.y, q.z)), 0.0) - rr< 0
@ti.func
def tri(r1, h, r2, round_unused, cone, vertex, p):
    r = vec3(p.x/r1, p.y, p.z/r2);rt=mix(1.0-cone,1.0,float(h-p.y)*0.5/h);r.z+=(r.x+1)*mix(-0.577, 0.577, vertex)
    q = ti.abs(r); return max(q.y-h,max(q.z*0.866025+r.x*0.5,-r.x)-0.5*rt)< 0
@ti.func
def make(func: ti.template(), p1, p2, p3, p4, p5, p6, pos, dir, up, color, mat, mode):
    max_r = 2 * int(max(p3,max(p1, p2))); dir = normalize(dir); up = normalize(cross(cross(dir, up), dir))
    for i,j,k in ti.ndrange((-max_r,max_r),(-max_r,max_r),(-max_r,max_r)):
        xyz = proj_plane(vec3(0.0,0.0,0.0), dir, up, vec3(i,j,k))
        if func(p1,p2,p3,p4,p5,p6,xyz):
            if mode == 0: scene.set_voxel(pos + vec3(i,j,k), mat, color) # additive
            if mode == 1: scene.set_voxel(pos + vec3(i,j,k), 0, color) # subtractive
            if mode == 2 and scene.get_voxel(pos + vec3(i,j,k))[0] > 0: scene.set_voxel(pos + vec3(i,j,k), mat, color)
@ti.kernel
def initialize_voxels():
    make(elli,32.0,28.1,38.0,0.0,0.0,0.0,vec3(0,-29,8),vec3(-0.6,0.1,0.8),vec3(0.1,1.0,0.0),rgb(197,197,197),1,0)
    make(elli,14.6,6.1,15.9,0.0,0.0,0.0,vec3(-19,-5,-8),vec3(-0.6,0.1,0.8),vec3(-0.2,0.9,-0.3),rgb(14,14,14),1,0)
    make(elli,14.6,6.1,15.9,0.0,0.0,0.0,vec3(25,-9,26),vec3(-0.6,0.1,0.8),vec3(0.6,0.7,0.4),rgb(14,14,14),1,0)
    make(elli,10.3,25.8,10.3,0.0,0.0,0.0,vec3(-26,-42,-11),vec3(-0.9,-0.1,-0.4),vec3(-0.0,1.0,-0.2),rgb(14,14,14),1,0)
    make(elli,2.6,5.5,2.6,0.0,0.0,0.0,vec3(-45,-32,-26),vec3(0.0,1.0,0.2),vec3(0.8,-0.1,0.6),rgb(14,14,14),1,0)
    make(elli,7.6,14.6,7.6,0.0,0.0,0.0,vec3(-45,-41,-25),vec3(-0.9,0.3,-0.4),vec3(0.3,0.9,0.1),rgb(14,14,14),1,0)
    make(elli,11.6,1.4,1.0,0.0,0.0,0.0,vec3(-24,-13,20),vec3(0.8,0.6,-0.0),vec3(-0.2,0.2,-1.0),rgb(14,14,14),1,0)
    make(elli,10.3,1.4,1.0,0.0,0.0,0.0,vec3(-3,-14,40),vec3(0.3,0.8,-0.5),vec3(-0.7,-0.1,-0.7),rgb(14,14,14),1,0)
    make(elli,3.7,3.6,1.0,0.0,0.0,0.0,vec3(-29,-19,16),vec3(0.3,0.8,-0.5),vec3(-0.7,-0.1,-0.7),rgb(14,14,14),1,0)
    make(elli,3.5,2.9,1.0,0.0,0.0,0.0,vec3(-6,-23,39),vec3(0.8,0.5,0.0),vec3(-0.4,0.7,-0.5),rgb(14,14,14),1,0)
    make(elli,5.7,1.4,1.0,0.0,0.0,0.0,vec3(-3,-20,41),vec3(0.3,0.8,-0.5),vec3(-0.7,-0.1,-0.7),rgb(14,14,14),1,0)
    make(elli,1.0,1.0,1.0,0.0,0.0,0.0,vec3(0,-20,44),vec3(0.3,0.8,-0.5),vec3(-0.7,-0.1,-0.7),rgb(14,14,14),1,0)
    make(elli,1.3,2.9,1.0,0.0,0.0,0.0,vec3(-22,-21,23),vec3(0.9,0.5,0.1),vec3(-0.4,0.8,-0.5),rgb(14,14,14),1,0)
    make(elli,5.7,1.4,1.0,0.0,0.0,0.0,vec3(-26,-18,19),vec3(0.6,0.8,-0.3),vec3(-0.7,0.2,-0.7),rgb(14,14,14),1,0)
    make(elli,3.0,1.4,1.0,0.0,0.0,0.0,vec3(-4,-24,41),vec3(0.3,0.8,-0.5),vec3(-0.7,-0.1,-0.7),rgb(14,14,14),1,0)
    make(elli,1.0,1.0,1.0,0.0,0.0,0.0,vec3(-2,-24,43),vec3(0.3,0.8,-0.5),vec3(-0.7,-0.1,-0.7),rgb(14,14,14),1,0)
    make(elli,5.7,1.4,1.0,0.0,0.0,0.0,vec3(-16,-34,30),vec3(0.5,0.8,-0.3),vec3(-0.6,0.1,-0.8),rgb(14,14,14),1,0)
    make(elli,8.3,1.4,1.0,0.0,0.0,0.0,vec3(-10,-45,31),vec3(0.4,0.8,-0.4),vec3(-0.7,-0.0,-0.7),rgb(14,14,14),1,0)
    make(elli,5.7,1.4,1.0,0.0,0.0,0.0,vec3(-26,-18,19),vec3(0.6,0.8,-0.3),vec3(-0.7,0.2,-0.7),rgb(14,14,14),1,0)
    make(elli,3.9,1.4,1.0,0.0,0.0,0.0,vec3(-24,-33,23),vec3(0.3,0.3,-0.9),vec3(0.2,0.9,0.3),rgb(158,158,158),1,0)
    make(elli,3.9,1.4,1.0,0.0,0.0,0.0,vec3(-4,-33,36),vec3(-0.8,-0.3,-0.5),vec3(-0.4,0.9,0.1),rgb(158,158,158),1,0)
    make(elli,3.9,1.4,1.0,0.0,0.0,0.0,vec3(-3,-36,37),vec3(-0.8,-0.3,-0.5),vec3(-0.4,0.9,0.1),rgb(158,158,158),1,0)
    make(elli,8.3,1.4,1.0,0.0,0.0,0.0,vec3(-22,-45,23),vec3(0.4,0.8,-0.4),vec3(-0.7,-0.0,-0.7),rgb(14,14,14),1,0)
initialize_voxels()
scene.finish()
