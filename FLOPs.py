class FLOP:
    """
Class containing the methods to calculate FLOPs for the convolution,
pooling and batch-norm operation for a convolution layer.

Initializing params:
    input_shape: (Tuple) The shape of the input feature maps with batch size as well. Format: (B,C,H,W)
    kernel_shape: (Tuple) The shape of the kernel feature maps with num_filters as well. Format: (N,C,K,K)
    stride: (Integer) The stride for the convolution operation. None for other operations
    padding: (Integer) The padding for the convolution operation
    dilation_rate: (Integer) The dilation rate for dilated convolution
    """

    input_shape = None
    filter_shape = None
    stride = None
    padding = None
    num_flops = float(0)
    dilation_rate = float(0)

    def __init__(self,input_shape,filter_shape,stride,padding, dilation_rate):
        super().__init__()
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding
        self.dilation_rate = dilation_rate

    def calculateFLOPs(self):
        """
    Prints the FLOPs for the ["Conv", "BN" , "Pool"] operations given
        """

        print('\nDilation Rate is: {}\n'.format(self.dilation_rate))

        if self.dilation_rate is not 0:
            self.filter_shape = (self.filter_shape[0],self.filter_shape[1],self.filter_shape[2]+2*self.dilation_rate,self.filter_shape[3]+2*self.dilation_rate)

        ## Simple/Dilated Convolution Operation 
        if self.filter_shape[1] == 0:
            num_operands = self.filter_shape[2] * self.filter_shape[3]
        else:
            num_operands = self.filter_shape[1] * self.filter_shape[2] * self.filter_shape[3]
        
        flops_per_instance_conv = 2*num_operands - 1 ## num_operands for multiplications and num_operands - 1 for additions
       
        num_instance_row = ((self.input_shape[2] - self.filter_shape[2] + 2*self.padding)/self.stride) + 1
        num_instance_column = ((self.input_shape[3] - self.filter_shape[3] + 2*self.padding)/self.stride) + 1
       
        num_instance = num_instance_column * num_instance_row * self.filter_shape[0] ## multiply by number of filters
       
        total_flops_single_input_conv = num_instance * flops_per_instance_conv
       
        total_batch_flops_conv = total_flops_single_input_conv * self.input_shape[0]

        self.num_flops += total_batch_flops_conv
       
        if total_batch_flops_conv / 1e9 > 1:   # for GFLOPs
            if self.dilation_rate is 0:
                print('Convolution: {} GFLOPs\n'.format(total_batch_flops_conv / 1e9 ))
            else:
                print('Dilated Convolution: {} GFLOPs\n'.format(total_batch_flops_conv / 1e9 ))
        else:
            if self.dilation_rate is 0:
                print('Convolution: {} MFLOPs\n'.format(total_batch_flops_conv / 1e6 ))
            else:
                print('Dilated Convolution: {} GFLOPs\n'.format(total_batch_flops_conv / 1e6 ))

        ## Pooling Operation
        total_flops_single_input_pool = num_instance
        total_batch_flops_pool = total_flops_single_input_pool * self.input_shape[0]

        self.num_flops += total_batch_flops_pool
       
        if total_batch_flops_pool / 1e9 > 1:   # for GFLOPs
            print('Pooling: {} GFLOPs\n'.format(total_batch_flops_pool / 1e9 ))
        else:
            print('Pooling: {} MFLOPs\n'.format(total_batch_flops_pool / 1e6 ))
        
        ## BN Operation
        total_flops_mean_bn = num_instance * (self.input_shape[0] - 1 + 1) ## number of additions + 1 division
        total_flops_std_bn = num_instance * (2*self.input_shape[0] + self.input_shape[0] - 1 + 1) ## subtract mean and square operations + additions + division
        total_batch_flops_bn = (total_flops_mean_bn + total_flops_std_bn) * self.input_shape[0]

        self.num_flops += total_batch_flops_bn
       
        if total_batch_flops_bn / 1e9 > 1:   # for GFLOPs
            print('Batch-Normalization: {} GFLOPs\n'.format(total_batch_flops_bn / 1e9 ))
        else:
            print('Batch-Normalization: {} MFLOPs\n'.format(total_batch_flops_bn / 1e6 ))
        
        ## RELU Activation
        num_flops_activation = num_instance_row * num_instance_column * self.filter_shape[0] * 2 ## For RELU activation, 1 comparison and 1 multiplication
        batch_flops_activation = num_flops_activation * self.input_shape[0]

        self.num_flops += batch_flops_activation

        if batch_flops_activation / 1e9 > 1:   # for GFLOPs
            print('RELU: {} GFLOPs\n'.format(batch_flops_activation / 1e9 ))
        else:
            print('RELU: {} MFLOPs\n'.format(batch_flops_activation / 1e6 ))

        ## Total FLOPs
        if self.num_flops / 1e9 > 1:   # for GFLOPs
            print('Total FLOPs: {} GFLOPs\n'.format(self.num_flops / 1e9 ))
        else:
            print('Total FLOPs: {} MFLOPs\n'.format(self.num_flops / 1e6 ))
        
        ## Percentage Contribution

        print('Operation           \t Percentage Contribution')
        print('------------------------------------------------')
        print('Convolution         \t {} %'.format(total_batch_flops_conv*100/self.num_flops))
        print('Pooling             \t {} %'.format(total_batch_flops_pool*100/self.num_flops))
        print('Batch-Normalization \t {} %'.format(total_batch_flops_bn*100/self.num_flops))
        print('RELU                \t {} %'.format(batch_flops_activation*100/self.num_flops))

if __name__ == '__main__':

    flops_calc = FLOP(input_shape=(1, 3 ,300 ,300), filter_shape=(64 ,3 ,3 ,3), stride=1, padding=1, dilation_rate=0)

    flops_calc.calculateFLOPs()
