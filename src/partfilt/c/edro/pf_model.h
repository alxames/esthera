#ifndef _PF_MODEL_H
#define _PF_MODEL_H 1

#define NUM_ANGLES 4

#define NUM_STATE_VARIABLES (NUM_ANGLES + 4)

typedef struct _particle_state
{
	float angles[NUM_ANGLES];
	float y;
	float z;
	float vY;
	float vZ;
	float weight;
}
particle_state;

typedef struct _control
{
}
control;

typedef struct _measurement
{
	float angles[NUM_ANGLES];
	float x;
	float y;
}
measurement;

#endif /* _PF_MODEL_H */
