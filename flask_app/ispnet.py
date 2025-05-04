import torch
import torch.nn as nn
import networks as N
class LiteISPNet(nn.Module):
	def __init__(self):
		super(LiteISPNet, self).__init__()
		# self.opt = opt
		ch_1 = 64
		ch_2 = 128
		ch_3 = 128
		n_blocks = 4
		# self.pre_ispnet_coord = opt.pre_ispnet_coord
		self.pre_ispnet_coord = False

		self.head = N.seq(
			N.conv(4, ch_1, mode='C')
		)  # shape: (N, ch_1, H/2, W/2)

		# if self.pre_ispnet_coord: # false during Training so commented out anyways
		# 	self.pre_coord = PreCoord(pre_train=True)

		self.down1 = N.seq(
			N.conv(ch_1+2, ch_1+2, mode='C'),
			N.RCAGroup(in_channels=ch_1+2, out_channels=ch_1+2, nb=n_blocks),
			N.conv(ch_1+2, ch_1, mode='C'),
			N.DWTForward(ch_1)
		)  # shape: (N, ch_1*4, H/4, W/4)

		self.down2 = N.seq(
			N.conv(ch_1*4, ch_1, mode='C'),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.DWTForward(ch_1)
		)  # shape: (N, ch_1*4, H/8, W/8)

		self.down3 = N.seq(
			N.conv(ch_1*4, ch_2, mode='C'),
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
			N.DWTForward(ch_2)
		)  # shape: (N, ch_2*4, H/16, W/16)

		self.middle = N.seq(
			N.conv(ch_2*4, ch_3, mode='C'),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
			N.conv(ch_3, ch_2*4, mode='C')
		)  # shape: (N, ch_2*4, H/16, W/16)

		self.up3 = N.seq(
			N.DWTInverse(ch_2*4),
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
			N.conv(ch_2, ch_1*4, mode='C')
		)  # shape: (N, ch_1*4, H/8, W/8)

		self.up2 = N.seq(
			N.DWTInverse(ch_1*4),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1*4, mode='C')
		)  # shape: (N, ch_1*4, H/4, W/4)

		self.up1 = N.seq(
			N.DWTInverse(ch_1*4),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1, mode='C')
		)  # shape: (N, ch_1, H/2, W/2)

		self.tail = N.seq(
			N.conv(ch_1, ch_1*4, mode='C'),
			nn.PixelShuffle(upscale_factor=2),
			N.conv(ch_1, 3, mode='C')
		)  # shape: (N, 3, H, W)   

	def forward(self, raw, coord=None):
		# input = raw
		input = torch.pow(raw, 1/2.2)
		h = self.head(input)
		if self.pre_ispnet_coord:
			pre_coord = self.pre_coord(raw) * 0.1
			pre_coord = torch.clamp(pre_coord, -1, 1)
			pre_coord = pre_coord.unsqueeze(dim=2).unsqueeze(dim=3)
			pre_coord = pre_coord + coord
			h_coord = torch.cat((h, pre_coord), 1)
		else:
			h_coord = torch.cat((h, coord), 1)
		
		d1 = self.down1(h_coord)
		d2 = self.down2(d1)
		d3 = self.down3(d2)
		m = self.middle(d3) + d3
		u3 = self.up3(m) + d2
		u2 = self.up2(u3) + d1
		u1 = self.up1(u2) + h
		out = self.tail(u1)

		return out
