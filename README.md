## Recon-Best Sigil Recognition

Each sigil is composed of multiple images: **recon** and **best**.
The image drawn by the player is called *test*.

We compare *test* with **recon**, which produces a completely white image without pixels.
If there are black pixels, then the image does not match **recon**.

We then compare *test* with **best**, which produces an image with black pixels.
We count the difference in black pixels between those of the difference and the total black pixels of *test*.
This allows us to get a percentage of similarity between *test* and **best**.

We also perform the inverse difference between **best** and *test*.
If the pixel difference is greater than 50%, then the sigil is incomplete.
