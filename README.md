# il-by-rl

A reference implementation of a reduction from imitation learning to reinforcement learning, presented in the following paper:

> Kamil Ciosek [Imitation Learning by Reinforcement Learning](https://openreview.net/forum?id=1zwleytEpYx), ICLR 2022.


## Getting Started

The implementation was tested on Python 3.9. To run the code, you need to install packages from `requirements.txt`. Using this repository requires the `git-lfs` extension. See [here](https://git-lfs.github.com/) for installation instructions.

To get started, simply follow these steps:

- Clone the repo locally with: `git clone
  https://github.com/spotify-research/il-by-rl.git`
- Move to the repository with: `cd il-by-rl`
- install the dependencies: `pip install -r requirements.txt`

## Running the experiments
Since running the experiments is computationally expensive, we provide pre-computed logs in the `sample-logs` directory. 
These can be plotted using the `plots.ipynb` notebook. Due to minor changes to the code, these logs are not absolutely 
identical to the ones used for the paper, but they support the exact same qualitative conclusion (ILR is as good as other
methods while being simpler).

If you want to re-run the experiments (regenerating the logs), you can run the command `python train.py --env=ENV --method=METHOD`, 
where `ENV` is one of `hopper, ant, walker, halfcheetah` and `METHOD` is one of `bc, il, gail, gmmil, sqil`. The new logs 
will be saved to the current directory.


## Support

Create a [new issue](https://github.com/spotify-research/il-by-rl/issues/new)


## Contributing

We feel that a welcoming community is important and we ask that you follow Spotify's
[Open Source Code of Conduct](https://github.com/spotify/code-of-conduct/blob/master/code-of-conduct.md)
in all interactions with the community.


## Authors

- [Kamil Ciosek](mailto:kamilc@spotify.com)

Follow [@SpotifyResearch](https://twitter.com/SpotifyResearch) on Twitter for
updates.


## License

Copyright 2022 Spotify, Inc.

Licensed under the Apache License, Version 2.0:
https://www.apache.org/licenses/LICENSE-2.0


## Security Issues?

Please report sensitive security issues via Spotify's bug-bounty program
(https://hackerone.com/spotify) rather than GitHub.
