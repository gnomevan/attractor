# When Does Training Start to Dream?
## How a Neural Network Crosses from Order into Chaos — and Why It Happens Sooner Than Anyone Thought

---

You are standing on the surface of the Earth. You feel still. But you are tracing a shape through space that you have never seen.

The Earth rotates once a day. That's one circle. While it rotates, it revolves around the sun once a year. That's a circle within a circle. The path you trace through space is not a loop — it is a helix wrapped around a donut. The shape is called a torus.

A torus is what you get whenever one cycle is embedded within another. This is not a metaphor. It is geometry. And it shows up everywhere that things cycle within cycling — which, if you look closely, is everywhere.

This paper is about a place where the donut breaks.

---

## The Experiment

Artificial neural networks learn by a process called gradient descent. You show the network data. It makes a prediction. You measure how wrong it was. Then you adjust its internal settings — millions of numerical dials called weights — to make it slightly less wrong. Repeat. Thousands of times. Eventually the network gets good at the task.

The size of each adjustment is controlled by a single number: the learning rate. Too small and the network learns painfully slowly. Too large and the adjustments overshoot — the network gets worse instead of better. Somewhere in the middle, learning works.

Here is the experiment we ran.

We took a neural network — a specific, simple one, with about 157,000 adjustable weights. We initialized those weights randomly. Then we made an exact copy and changed the copy's weights by an absurdly small amount — one part in a hundred million. A perturbation so small that no instrument could detect it. No physical process could distinguish the original from the copy.

Then we trained both networks on the same data, with the same learning rate, using the same procedure. Everything identical except for that immeasurable nudge in the starting weights.

And we asked: do the two networks learn the same thing?

At low learning rates, they do. The nudge washes out. Both networks converge to the same function. It doesn't matter where you started — you end up in the same place.

At high learning rates, they don't. The nudge amplifies. The two networks, despite identical training, learn measurably different things from the same experience. The difference between them grows exponentially — doubles, doubles again, doubles again — until their outputs are as different as if they'd been trained on different data entirely.

Somewhere between "low" and "high," there is a boundary. Below it, training is deterministic — the outcome depends on the data, not on the starting conditions. Above it, training is chaotic — the outcome depends on differences too small to measure.

We found that boundary. And it is not where anyone expected it.

---

## The Edge of Stability

In 2021, a group of researchers at Carnegie Mellon discovered something surprising about how neural networks train. They watched a quantity called sharpness — a measure of how curved the loss landscape is beneath the network's feet. Think of it as the steepness of the terrain the network is navigating.

Classical optimization theory says: if the terrain is too steep relative to your step size, you'll overshoot. You'll step past the valley floor, land on the opposite slope, and oscillate wildly. The math gives a precise threshold — a ratio between step size and steepness — beyond which the system should diverge.

What Cohen and colleagues found is that during training, the steepness rises until it hits exactly that threshold — then balances there. The loss wobbles but doesn't blow up. The network keeps learning, just not smoothly. They called this the Edge of Stability.

The Edge of Stability is the curvature limit. It's where the landscape pushes back against the step size. It was natural to assume this was also where chaos begins — where training starts to depend on its initial conditions.

It isn't.

---

## The Finding

The boundary where chaos begins — where those two nearly-identical networks start learning different things — sits at about **6.6% of the Edge of Stability**.

Not near the Edge. Not approaching it. Six and a half percent of it.

If the Edge of Stability is a cliff, chaos starts while you're still in the parking lot.

This means that at most learning rates anyone would ever use in practice, training is already chaotic. Two runs with different random initializations will produce genuinely different networks — not because of noise, not because of randomness in the data, but because the dynamics of learning itself are sensitive to conditions too small to observe.

---

## What Does "Chaos" Mean Here?

Chaos has a precise meaning in mathematics, and it's not the colloquial sense of "random" or "disordered."

A chaotic system is deterministic — there's no randomness involved. Given the same starting point, it will always follow the same path. But it is sensitive to initial conditions: two starting points that are immeasurably close will, over time, end up in completely different places. The divergence is exponential — it doesn't grow gradually. It doubles, and doubles again, and again.

The weather is chaotic. The equations governing air flow are deterministic. But a butterfly's wingbeat — a perturbation too small to measure — can, over weeks, change where a storm makes landfall. This isn't poetry. It's a mathematical theorem.

The quantity that measures this is called the Lyapunov exponent. When it's negative, nearby starting points converge — small differences wash out. When it's positive, they diverge — small differences amplify without limit. The zero-crossing of the Lyapunov exponent is the precise boundary between order and chaos.

That's what we measured. And we measured it in "function space" — not tracking whether the networks' internal weights diverge (which can happen for boring reasons, like symmetries in how weights are arranged) but whether the networks' actual outputs diverge. Whether they learn different functions.

---

## The Shape of Order: The Torus

Here is where it gets geometric.

When a system is orderly — cycling, but not chaotic — its behavior traces a shape. If it has one frequency (like a pendulum), the shape is a circle. If it has two incommensurate frequencies (like a wobbling orbit), the shape is a torus — a donut. The path winds around the donut's surface, never exactly repeating, but staying on the surface forever. This is called quasiperiodic motion.

Imagine training a neural network at a low learning rate. The loss goes down, oscillates a little, goes down more. The sharpness oscillates. If you could see the training trajectory in the space of all possible network behaviors, it would trace something like a path on a torus — winding, quasiperiodic, orderly.

The torus is the geometry of a system with nested cycles that hasn't broken yet.

---

## The Shape of Chaos: The Strange Attractor

Now increase the learning rate.

The torus begins to distort. Some of its structure survives — the robust parts, the ones with the most irrational frequency ratios. Other parts shatter. They don't disappear; they fragment into something called Cantori — fractal dust, the ghosts of the torus's surface, riddled with gaps.

Increase the learning rate further, and the torus is replaced entirely. In its place is a strange attractor — a fractal object. The training trajectory is still bounded (the network doesn't diverge to infinity), but it never repeats. It winds through a structure that has fractional dimension — not a surface, not a volume, but something in between. And two trajectories that start near each other on the strange attractor diverge exponentially.

This is what a positive Lyapunov exponent looks like in geometric language. The donut has broken. What's left is a fractal. And the network's fate depends on exactly where on that fractal it started.

---

## Three Mathematicians Predicted This

The transition from torus to strange attractor is not something we discovered. It was predicted in 1971 by David Ruelle and Floris Takens, who were trying to understand turbulence — how a smooth laminar flow breaks into chaotic turbulence.

Their theorem describes a sequence:

**Stillness** — the system is at rest. A ball at the bottom of a bowl.

**Oscillation** — the system cycles. One frequency. A swinging pendulum. In training, this would be the loss going up and down rhythmically.

**Quasiperiodicity** — the system has two or three incommensurate frequencies. It traces a torus. In training, the loss oscillates at one frequency while the sharpness oscillates at another.

**Chaos** — the torus breaks. It is replaced by a strange attractor. The system is bounded but unpredictable. In training, the network still learns, but what it learns depends on where it started.

Ruelle and Takens proved something remarkable: **you only need three nested frequencies for the torus to be structurally unstable.** Any perturbation, no matter how small, can destroy it and replace it with a strange attractor. Three layers of cycling is enough for the geometry of order to shatter into the geometry of chaos.

A neural network with multiple layers, trained on complex data, has far more than three things oscillating at once. The surprise is not that chaos appears. The surprise is that it takes as long as it does.

---

## The KAM Mosaic

There's a companion theorem from the 1950s and 60s, proved by Kolmogorov, Arnold, and Moser. It says that when you perturb an ordered system, not all the order is destroyed at once.

Some tori survive — the ones whose frequency ratios are hard to approximate by simple fractions. They're the irrational holdouts, the ones that resist resonance. They persist, slightly warped, like islands of stability in an ocean of chaos.

Other tori — the ones near resonance, whose frequencies are close to simple ratios — are destroyed. They fragment into fractal remnants.

The result is a mosaic: islands of order surrounded by seas of chaos, with smaller islands inside the seas, and smaller seas inside the islands, at every scale. It looks like a fractal because it is one.

This is exactly what we see in the transition zone of our experiment. At learning rates near the chaos boundary, some random seeds produce ordered behavior and others produce chaotic behavior. The outcome depends on which island or which sea the initialization happens to land on. The boundary between order and chaos is not a clean line. It is a fractal coastline — the closer you look, the more structure appears.

---

## Why This Matters

### Training is less predictable than we thought.

If chaos begins at 6.6% of the stability threshold, then virtually every practical learning rate puts training in the chaotic regime. Two runs of the same network on the same data, differing only in random initialization, will produce genuinely different results. This isn't a quirk. It's a dynamical phase.

### The Edge of Stability isn't the edge of anything.

The Edge of Stability is where the loss landscape's curvature catches up with the step size. It's an important phenomenon — but it's not where order ends and chaos begins. By the time the system reaches the Edge of Stability, it has been chaotic for a long time. The Edge of Stability is a curvature event happening inside a chaos that was already underway.

### Chaos might be why training works.

This sounds paradoxical, but chaotic dynamics have a feature that ordered dynamics don't: they explore. A chaotic trajectory wanders widely through the space of possible network behaviors, visiting regions that an ordered trajectory would never reach. This sensitivity to initial conditions means that different starting points explore different parts of the landscape — and the best solutions may only be found by trajectories that wander chaotically.

If this is right, then the chaos isn't a bug. It's the mechanism by which gradient descent avoids getting trapped in bad solutions. The donut has to break for the network to learn well.

### Ensembles get diversity for free.

A common technique in machine learning is to train several networks and average their predictions. This works better than any single network because the different networks make different mistakes. But why are they different? They were trained on the same data.

The answer may be chaos. If training dynamics are chaotic above η_c, then different random initializations automatically produce diverse networks. You don't need to engineer diversity. The dynamics hand it to you.

---

## What We Haven't Seen Yet

We have measured where the transition happens. We haven't seen it.

To actually see the torus breaking, we would need to:

**Build a bifurcation diagram.** Train the network at hundreds of densely-spaced learning rates and record the loss at convergence. If the transition follows the classic route, we should see the single converged loss value split into two alternating values, then four, then eight, then a chaotic cloud — the same pattern that appears in dripping faucets, population dynamics, and fluid turbulence. It has a name: the Feigenbaum cascade. And it has universal constants that are the same across all systems that exhibit it.

**Listen to the frequencies.** Record the training loss at every step and compute its frequency content. Below the chaos boundary, expect a few clean frequencies — the signature of motion on a torus. Above it, expect broadband noise — the signature of a strange attractor. The transition from peaks to noise is the sound of the donut breaking.

**Reconstruct the attractor.** Using a technique called Takens embedding, you can take a single time series (the training loss) and unfold it into a higher-dimensional space where the underlying geometric structure becomes visible. Below the boundary, the shape should look like a torus. Above it, like a fractal. This would be the most direct evidence: a picture of the geometry changing.

These experiments are computationally straightforward. They are the next step.

---

## The Bigger Picture

This paper measures one thing: the learning rate at which gradient descent becomes chaotic. But the finding connects to a much larger pattern.

Wherever cycles nest within cycles, toroidal geometry emerges. Your heart beats within your breath. Your breath cycles within your day. Your day cycles within the seasons. Each faster rhythm is modulated by the slower rhythm containing it. The geometry of all this nesting is toroidal.

And wherever toroidal dynamics are stressed — perturbed, coupled too strongly, nested too deeply — the torus breaks and fractal complexity appears. The branching of your lungs, the forking of rivers, the crumpling of coastlines — these are the spatial signatures of temporal processes whose toroidal structure has been pushed into chaos.

Gradient descent does the same thing. At low learning rates, the dynamics are toroidal — orderly, quasiperiodic, reproducible. At higher learning rates, the torus shatters and a strange attractor takes its place — chaotic, fractal, sensitive to initial conditions. The transition happens through the same mathematical mechanisms — KAM theory, Ruelle-Takens, Feigenbaum period-doubling — that govern the transition from laminar to turbulent flow.

The learning rate is the dial that turns one geometry into the other. We found where it turns.

---

*The torus is what training looks like before it starts to improvise. The strange attractor is what it looks like after. Between them is a fractal boundary — not a line but a coastline, getting more complex the closer you look. That boundary sits at about 6.6% of where we thought it was.*

*The donut breaks early. And what comes after is not disorder. It is a different kind of order — wilder, more complex, and perhaps more capable than the smooth geometry it replaced.*
