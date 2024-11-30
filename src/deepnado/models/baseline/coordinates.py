"""
CoordConv for tornado detection
"""

import torch

from torch.nn import Module, Conv2d, ReLU


class CoordConv2D(Module):
    """
    CoordConv2D layer for working with polar data.

    This module takes a tuple of inputs (image tensor,   image coordinates)
    where,
        image tensor is [batch, in_image_channels, height, width]
        image coordinates is [batch, in_coord_channels, height, width]

    This returns a tuple containing the CoordConv convolution and
    a (possibly downsampled) copy of the coordinate tensor.


    """

    def __init__(
        self,
        in_image_channels,
        in_coord_channels,
        out_channels,
        kernel_size,
        padding="same",
        stride=1,
        activation="relu",
        **kwargs,
    ):
        super(CoordConv2D, self).__init__()
        self.n_coord_channels = in_coord_channels
        self.conv = Conv2d(
            in_image_channels + in_coord_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            **kwargs,
        )
        self.strd = stride
        self.padding = padding
        self.ksize = kernel_size
        if activation is None:
            self.conv_activation = None
        elif activation == "relu":
            self.conv_activation = ReLU()
        else:
            raise NotImplementedError("activation %s not implemented" % activation)

    def forward(self, inputs):
        """
        inputs is a tuple containing
          (image tensor,   image coordinates)

        image tensor is [batch, in_image_channels, height, width]
        image coordinates is [batch, in_coord_channels, height, width]

        """
        x, coords = inputs
        x = torch.cat((x, coords), axis=1)
        x = self.conv(x)

        # only apply activation to conv output
        if self.conv_activation:
            x = self.conv_activation(x)

        # also return coordinates
        if self.padding == "same" and self.strd > 1:
            coords = coords[..., :: self.strd, :: self.strd]
        elif self.padding == "valid":
            # If valid padding,  need to start slightly off the corner
            i0 = self.ksize[0] // 2
            if i0 > 0:
                coords = coords[..., i0 : -i0 : self.strd, i0 : -i0 : self.strd]  # noqa: E203
            else:
                coords = coords[..., :: self.strd, :: self.strd]

        return x, coords
