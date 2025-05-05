#pragma once
#include "vectors.h"


class Particle {
    vec2 position;
    float radius;
    vec2 velocity;
    float mass;
    float inverseMass;
    vec2 acceleration;
    uchar4 color;
    bool active = true;
    bool selected = false;

    vec2 previousPosition;
    vec2 pointInCircle;
    float angularVelocity;
    vec2 externalForce;
    vec2 forcesSum;
    int parentIndex;
    
    
    
    __host__ __device__
    Particle(vec2 position, float mass, float radius, uchar4 color, int index) 
        : position(position), radius(radius), mass(mass), color(color), parentIndex(index) {

        inverseMass = (mass > 0) ? 1.0 / mass : 0.0f;  // avoid division by zero
        velocity = vec2(0.f,0.f); // initial velocity
        acceleration = vec2(0.0f, 0.0f);
    }
    // integrate acceleration -> velocity -> position
    void eulerIntegration(float dt) {
        acceleration = forcesSum * inverseMass;
        velocity += acceleration * dt;
        position += velocity * dt;
        forcesSum = vec2(0,0);  // clear forces
    }

    // integrate with Verlet
    void verletIntegration(float dt, const vec2 &newAcceleration) {
        position += velocity * dt + acceleration * 0.5f * dt * dt;
        velocity += (acceleration + newAcceleration) * 0.5f * dt;
        acceleration = newAcceleration;

    }

    void verletIntegration(float dt) {
        // Verlet without velocity: uses current and previous position
        vec2 temp = position;
        position = position +  ( position - previousPosition) +  acceleration * dt * dt;
        previousPosition = temp;

        acceleration = vec2(0.0f); // reset for next frame
    }


    __host__ __device__
    void applyForce(vec2 f) {
        vec2 force = f;
        forcesSum += force;
    }

    void applyPositionalCorrection(vec2& otherPos, float targetDistance, float strength = 1.0f) {
        vec2 delta = otherPos - position;
        float dist = delta.mag();
    
        if (dist < 1e-5f) return; // avoid div by 0
    
        vec2 dir = delta / dist;
        float correction = (dist - targetDistance) * 0.5f * strength;
    
        // move both particles toward rest distance
        position += dir * correction;
        otherPos -= dir * correction;
    }
    

};
