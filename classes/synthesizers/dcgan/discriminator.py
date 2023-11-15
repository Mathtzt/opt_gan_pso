import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_conv_blocks: int, nc: int = 3, ndf: int = 64):
        super(Discriminator, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.nc = nc
        self.ndf = ndf

        self.arquitecture_values = self.__get_arquitectures(n_conv_blocks)
    
        self.conv_blocks = nn.ModuleList()

        # Adiciona camadas convolucionais à lista
        for iconv in range(0, self.n_conv_blocks - 1):
            if iconv == 0:
              self.conv_blocks.append(
                  nn.Sequential(
                      nn.Conv2d(self.nc, self.ndf * (2 ** iconv), self.arquitecture_values[iconv][0], self.arquitecture_values[iconv][1], self.arquitecture_values[iconv][2], bias=False),
                      nn.LeakyReLU(0.2, inplace=True),
                  )
              )
            else:
              self.conv_blocks.append(
                  nn.Sequential(
                      nn.Conv2d(self.ndf * (2 ** (iconv - 1)), self.ndf * (2 ** iconv), self.arquitecture_values[iconv][0], self.arquitecture_values[iconv][1], self.arquitecture_values[iconv][2], bias=False),
                      nn.BatchNorm2d(self.ndf * (2 ** iconv)),
                      nn.LeakyReLU(0.2, inplace=True)
                  )
              )

        self.output_conv_block = nn.Sequential(
                nn.Conv2d(self.ndf * (2 ** iconv), 1, self.arquitecture_values[-1][0], self.arquitecture_values[-1][1], self.arquitecture_values[-1][2], bias=False),
                nn.Sigmoid()
            )

    def forward(self, input):
        for conv_block in self.conv_blocks:
            input = conv_block(input)

        # Camada de saída
        output = self.output_conv_block(input)

        return output
    
    def __get_arquitectures(self, n_conv_blocks):
        # [kernel_size, stride, padding] #

        vdict_ = {
            2: [
               [20, 2, 1],   # 8 x 8 
               [8, 1, 0]   # 1 x 1
               ],
            3: [
               [12, 2, 1],   # 12 x 12
               [8, 2, 1],   # 4 x 4
               [4, 1, 0]   # 1 x 1
               ],
            4: [
               [4, 2, 1],   # 16 x 16
               [4, 2, 1],   # 8 x 8
               [4, 2, 1],   # 4 x 4
               [4, 1, 0]    # 1 x 1
               ],
            5: [
               [4, 2, 1],   # 16 x 16
               [4, 2, 1],   # 8 x 8
               [4, 2, 1],   # 4 x 4
               [4, 2, 1],   # 2 x 2
               [2, 1, 0]    # 1 x 1
               ],
            6: [
               [4, 2, 1],   # 16 x 16
               [4, 2, 1],   # 8 x 8
               [3, 1, 0],   # 6 x 6
               [3, 1, 0],   # 4 x 4
               [2, 2, 0],   # 2 x 2
               [2, 1, 0]    # 1 x 1
               ],
            7: [
               [4, 2, 1],   # 16 x 16
               [7, 1, 1],   # 12 x 12
               [5, 1, 0],   # 8 x 8
               [3, 1, 0],   # 6 x 6
               [2, 2, 1],   # 4 x 4
               [2, 2, 0],   # 2 x 2
               [2, 1, 0]    # 1 x 1
               ]
        }
    
        return vdict_.get(n_conv_blocks)