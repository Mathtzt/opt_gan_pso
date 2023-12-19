import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, n_conv_blocks: int, nc: int, nz: int, ngf: int = 64):
        super(Generator, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.nc = nc
        self.nz = nz
        self.ngf = ngf

        self.arquitecture_values = self.__get_arquitectures(n_conv_blocks)
        if self.n_conv_blocks > 1:

            self.conv_blocks = nn.ModuleList()
            # Adiciona camadas convolucionais à lista
            for i, iconv in enumerate(reversed(range(self.n_conv_blocks - 1))):
                if i == 0:
                    self.conv_blocks.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(self.nz, self.ngf * (2 ** iconv), self.arquitecture_values[i][0], self.arquitecture_values[i][1], self.arquitecture_values[i][2], bias=False),
                            nn.BatchNorm2d(self.ngf * (2 ** iconv)),
                            nn.ReLU(True)
                        )
                    )
                else:
                    self.conv_blocks.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(self.ngf * (2 ** (iconv + 1)), self.ngf * (2 ** (iconv)), self.arquitecture_values[i][0], self.arquitecture_values[i][1], self.arquitecture_values[i][2], bias=False),
                            nn.BatchNorm2d(self.ngf * (2 ** (iconv))),
                            nn.ReLU(True)
                        )
                    )

        ngf = self.ngf if self.n_conv_blocks > 1 else self.nz
        self.output_conv_block = nn.Sequential(
                nn.ConvTranspose2d(ngf, self.nc, self.arquitecture_values[self.n_conv_blocks - 1][0], self.arquitecture_values[self.n_conv_blocks - 1][1], self.arquitecture_values[self.n_conv_blocks - 1][2], bias=False),
                nn.Tanh()
            )

    def forward(self, input):
        input_ = input.clone()

        if self.n_conv_blocks > 1:
            for conv_block in self.conv_blocks:
                input_ = conv_block(input_)

        # Camada de saída
        output = self.output_conv_block(input_)

        return output
    
    def __get_arquitectures(self, n_conv_blocks):
        # [kernel_size, stride, padding] #

        vdict_ = {
            1: [
               [32, 1, 0]   # 32 x 32
               ],
            2: [
               [8, 1, 0],   # 8 x 8 
               [20, 2, 1]   # 32 x 32
               ],
            3: [
               [4, 1, 0],   # 4 x 4
               [8, 2, 1],   # 12 x 12
               [12, 2, 1]   # 32 x 32
               ],
            4: [
               [4, 1, 0],   # 4 x 4
               [4, 2, 1],   # 8 x 8
               [4, 2, 1],   # 16 x 16
               [4, 2, 1]    # 32 x 32
               ],
            5: [
               [2, 1, 0],   # 2 x 2
               [4, 2, 1],   # 4 x 4
               [4, 2, 1],   # 8 x 8
               [4, 2, 1],   # 16 x 16
               [4, 2, 1]    # 32 x 32
               ],
            6: [
               [2, 1, 0],   # 2 x 2
               [2, 2, 0],   # 4 x 4
               [3, 1, 0],   # 6 x 6
               [3, 1, 0],   # 8 x 8
               [4, 2, 1],   # 16 x 16
               [4, 2, 1]    # 32 x 32
               ],
            7: [
               [2, 1, 0],   # 2 x 2
               [2, 2, 0],   # 4 x 4
               [2, 2, 1],   # 6 x 6
               [3, 1, 0],   # 8 x 8
               [5, 1, 0],   # 12 x 12
               [7, 1, 1],   # 16 x 16
               [4, 2, 1]    # 32 x 32
               ]
        }
    
        return vdict_.get(n_conv_blocks)