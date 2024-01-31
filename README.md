# PDE Animation

## Getting Started

```
git clone https://github.com/graysoncroom/PDEAnimation.git
cd PDEAnimation
python -m venv env
source ./env/bin/activate
pip install manim scipy
```

Now you'll be in a python virtual environment with the manim and scipy libraries installed.

Note that you'll need to source `./env/bin/activate` every time you close your terminal
and want to start working on the code again.

## Rendering the animation(s)

Example:
```
manim -pqh src/matchsticks.py PDEAnimation
```

The `-p` option tells manim to open the animation in a video player after
the rendering completes (stands for play).

The `-qh` option tells manim to render with high quality. 

You could also give it `-ql` instead as your iterating on an animation and want the rendering to go faster.

`src/matchsticks.py` is the source file of your animation code

`PDEAnimation` is the class name of the manim class that describes the animation

Note: Whether you pass manim the `-p` option or not, the rendered video will be
stored in the `./media` directory. If a `./media` directory doesn't exist, it
will be created for you.

## Problems and Solutions

If manim is complaining about libraries, check out the [Manim Documentation](https://docs.manim.community/en/stable/installation/linux.html)

It's likely you'll need to install a few system-level libs (e.g. ffmpeg, libpango1, libcairo2, ...)
