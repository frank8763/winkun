"""New declared sparse objective plus terminal-consistent potential shaping."""
import math

GAMMA=math.exp(-.002/60.)

def reward(previous_distance,next_distance,goal,collision,terminal):
    assert not (goal and collision)
    base=-.01*.002+10.*bool(goal)-10.*bool(collision)
    old_phi=-.3*previous_distance
    new_phi=0. if terminal else -.3*next_distance
    return base+GAMMA*new_phi-old_phi

def checks():
    # Shaping telescopes through a nonterminal block and sets the absorbing potential to zero.
    d=[3.,2.5,2.];a=reward(d[0],d[1],False,False,False);b=reward(d[1],d[2],True,False,True)
    base=-.00002+GAMMA*(10.-.00002)
    assert abs(a+GAMMA*b-base-.9)<1e-10
    assert abs(GAMMA**25-math.exp(-.05/60.))<1e-10
    # A time truncation must retain the nonterminal potential and bootstrap.
    assert reward(2.,2.,False,False,False)!=reward(2.,2.,False,False,True)
