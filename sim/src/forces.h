#pragma once
#include "particles.h"

class Force {

    
    public: 

    static Vec2 friction(Particle &p, const float K) {
        Vec2 frictionDirection = p.velocity.unitVector() * -1;
        return frictionDirection * K * PIXELS_PER_METER; // * scaleFactor;
       
    }

    static Vec2 weight(Particle &p, float K=9.8) {
        Vec2 weight = Vec2(0, K * PIXELS_PER_METER);// * scaleFactor;
        return weight / p.inverseMass;
    }
    
    /**
     * Param: Particle  
     * Param: K   is the drag coefficient
     */
    static Vec2 drag(Particle &p, const float K)
        {
            // calculate drag direction (inverse)
            Vec2 dragDir = p.velocity.unitVector() * -1;
            // Calculate the drag magnitude K * |v|^2
            float dragMag =p.velocity.magSquared() * K;
            
            return dragMag > 0 ? dragDir * dragMag : Vec2(0,0);
        }
    
    static Vec2 impulse(Particle&p, Vec2 origin, const float K) {
        Vec2 direction = (p.position - origin).unitVector();
        float magnitud = (p.position - origin).mag() * K;
        return  direction * (magnitud * p.inverseMass);
    }

    static Vec2 generalGravitation(Particle &a, Particle &b, float G) {

        /*
            * Fg = G * (Ma * Mb)/ |d|^2 * direction(unitvector)
        */

        /* Since this is not real life, adding a small epsilon value to prevent 
            * singularity when objects are very close (avoid unpredictable shoot-outs)
        */

        float epsilon = 0.0f;
        float distanceSquared    = (b.position - a.position).magSquared() + epsilon;
        Vec2 attractionDirection = (b.position - a.position).unitVector();
        float attractionMagnitude = G * (a.mass * b.mass) / distanceSquared;
        Vec2 attractionForce = attractionDirection * attractionMagnitude;

        return attractionForce ;

    }

    static Vec2 generalGravitation(Particle &a, Particle &b, float G, float minDSq, float maxDSq) {

        /*
         * Fg = G * (Ma * Mb)/ |d|^2 * direction(unitvector)
        */

        /* Since this is not real life, adding a small epsilon value to prevent 
         * singularity when objects are very close (avoid unpredictable shoot-outs)
        */

        // float epsilon = 1.0f;
        float distanceSquared    = (b.position - a.position).magSquared(); // + epsilon;
        distanceSquared = clip(distanceSquared, minDSq, maxDSq);
        Vec2 attractionDirection = (b.position - a.position).unitVector();
        float attractionMagnitude = G * (a.mass * b.mass) / distanceSquared;
        Vec2 attractionForce = attractionDirection * attractionMagnitude;

        return attractionForce * PIXELS_PER_METER * scaleFactor;

    }
    

    static Vec2 spring(Particle &p, Vec2 anchor, float restLength, const float K){
        /*
        * Fs = -k * dxl       dx = A or delta or displament
        */
        // * Calculate the distance between the anchor and the object
        Vec2 delta = p.position - anchor;
        float length = delta.mag();
        // Prevent divide by zero / normalize instability
        if (length < 1e-6f) return Vec2(0.0f, 0.0f);
        // * Find the spring displament considering the rest length
        float displament = length - restLength;
        // * Calculate the direction and the magnitude of the spring force
        Vec2 springDirection = delta.unitVector();
        float springMagnitude = -K * displament;

        // Calculate the final resulting spring force vector
        Vec2 springForce = springDirection * springMagnitude;
    
        return springForce;

    }

    static Vec2 spring(Particle &a, Particle &b, float restLength, const float K){
        /*
        * Fs = -k * dxl       dx = A or delta or displament
        */
        // * Calculate the distance between the anchor and the object
        Vec2 delta = a.position - b.position;
        float length = delta.mag();
        // Prevent divide by zero / normalize instability
        if (length < 1e-6f) return Vec2(0.0f, 0.0f);
       
        
        // * Find the spring displament considering the rest length
        float displament = length - restLength;
        // // Clamp x to avoid runaway forces
        // const float maxStretch = 100.0f;  // tweak based on system
        // displament = clip(length, -maxStretch, maxStretch);
        
        // * Calculate the direction and the magnitude of the spring force
        Vec2 springDirection = delta.unitVector();
        float springMagnitude = -K * displament;

        // Calculate the final resulting spring force vector
        Vec2 springForce = springDirection * springMagnitude;
        // static int counter = 0;
        // // Optionally clamp final force too
        // const float maxForce = 10000.0f;
        // if (springForce.mag() > maxForce ) {
        //     springForce = springForce.normalize() * maxForce;
        //     printf("FIXED: %d\n", counter++);
        // }
        // springForce.log("force");
        return springForce;

    }

    static Vec2 dampedSpring(Particle &a, Particle &b, float restLength, const float K, const float damping) {
        Vec2 delta = a.position - b.position;
        float length = delta.mag();
    
        if (length < 1e-6f) return Vec2(0.0f); // prevent division by zero
    
        Vec2 springDirection = delta / length;
        float displacement = length - restLength;
                
        // printf("displacement %f\n", displacement);
        // Spring force: F = -k * x
        float springMagnitude = -K * displacement-1.0;
    
        // Relative velocity projected onto the spring direction
        Vec2 relVel = a.velocity - b.velocity;
        float dampingMagnitude = damping * relVel.dot(springDirection);
    
        // Total force
        float totalMagnitude = springMagnitude - dampingMagnitude;
    
        return springDirection * totalMagnitude;
    }

    static Vec2 dampedSpring(Particle &a, Vec2 anchor, float restLength, const float K, const float damping) {
        Vec2 delta = a.position - anchor;
        float length = delta.mag();
    
        if (length < 1e-6f) return Vec2(0.0f); // prevent division by zero
    
        Vec2 springDirection = delta / length;
        float displacement = length - restLength;
                
        // printf("displacement %f\n", displacement);
        // Spring force: F = -k * x
        float springMagnitude = -K * displacement-1.0;
    
        // Relative velocity projected onto the spring direction
        Vec2 relVel = a.velocity - anchor;
        float dampingMagnitude = damping * relVel.dot(springDirection);
    
        // Total force
        float totalMagnitude = springMagnitude - dampingMagnitude;
    
        return springDirection * totalMagnitude;
    }


    static Vec2 dampedSpringNoStretch(Particle &a, Particle &b, float restLength, float K, float damping) {
        Vec2 delta = a.position - b.position;
        float length = delta.mag();
    
        if (length < 1e-6f) return Vec2(0.0f); // avoid div by zero
        // printf("length: %f\n", length);
        // Only apply force when compressed (shorter than restLength)
        // length = fmin(length, restLength);
    
        Vec2 springDir = delta / length;
        float displacement = length - restLength; // this is ≤ 0
    
        float springForce = -K * displacement;
        Vec2 relativeVel = a.velocity - b.velocity;
        float dampingForce = damping * relativeVel.dot(springDir);
    
        float totalForce = springForce - dampingForce;
        return springDir * totalForce;
    }
    
    


        
};