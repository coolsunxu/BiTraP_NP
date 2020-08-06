
import torch
import torch.nn as nn
import torch.nn.functional as F

# 先验网络
class Prior(nn.Module):
	def __init__(self,input_size=256,output_size=64):
		super(Prior,self).__init__()
		self.input_size = input_size # 输入大小
		self.output_size = output_size # 输出大小
		
		self.fc1 = nn.Linear(input_size,128)
		self.fc2 = nn.Linear(128,64)
		self.fc3 = nn.Linear(64,output_size)
		
	def forward(self,x):
		# x：[bs,256]
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

# 后验网络
class Recognition(nn.Module):
	def __init__(self,input_size=512,output_size=64):
		super(Recognition,self).__init__()
		self.input_size = input_size # 输入大小
		self.output_size = output_size # 输出大小
		
		self.fc1 = nn.Linear(input_size,256)
		self.fc2 = nn.Linear(256,128)
		self.fc3 = nn.Linear(128,output_size)
		
	def forward(self,x):
		# x：[bs,512]
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
		
# 目标网络
class Goal(nn.Module):
	def __init__(self,input_size=256+32,output_size=2):
		super(Goal,self).__init__()
		self.input_size = input_size # 输入大小
		self.output_size = output_size # 输出大小
		
		self.fc1 = nn.Linear(input_size,128)
		self.fc2 = nn.Linear(128,64)
		self.fc3 = nn.Linear(64,output_size)
		
	def forward(self,x):
		# x：[bs,256+32]
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
			
		

class BiTrap(nn.Module):
	def __init__(self,embedding_size=64,input_size=2,output_size=2,
			gru_size=256,latent_size=32,gru_input_size=64,num_layer=1,
			pre_len=12
		):
		super(BiTrap,self).__init__()
		self.input_size = input_size # 输入大小
		self.output_size = output_size # 输出大小
		self.embedding_size = embedding_size # 空间编码
		self.gru_size = gru_size # GRU隐藏层大小
		self.latent_size = latent_size # z的大小
		self.gru_input_size = gru_input_size # GRU输入大小
		self.num_layer = num_layer # GRU层数
		self.pre_len = pre_len # 预测长度
		
		# 观测轨迹升维
		self.fcx = nn.Linear(input_size,embedding_size)
		# 预测轨迹升维
		self.fcy = nn.Linear(input_size,embedding_size)
		# 观测轨迹GRU
		self.grux = nn.GRU(embedding_size,gru_size)
		# 预测轨迹GRU
		self.gruy = nn.GRU(embedding_size,gru_size)
		
		# 先验网络 后验网络 目标网络
		self.prior = Prior(gru_size,2*latent_size)
		self.recognition = Recognition(gru_size*2,2*latent_size)
		self.goal = Goal(gru_size+latent_size,output_size)
		
		# 状态转换网络
		self.fcf = nn.Linear(gru_size+latent_size,gru_input_size)
		self.fc2 = nn.Linear(gru_size+latent_size,gru_size)
		self.fc3 = nn.Linear(gru_size,gru_input_size)
		
		self.fc4 = nn.Linear(gru_size+latent_size,gru_size)
		self.fc5 = nn.Linear(output_size,gru_input_size)
		
		# GRUcell
		self.forward_gru = nn.GRUCell(gru_input_size,gru_size)
		self.backward_gru = nn.GRUCell(gru_input_size,gru_size)
		
		# 最后的输出
		self.fc6 = nn.Linear(2*gru_size,output_size)
		
	def forward(self,x,mode='train',y=None):
		n,s = x.shape[-1],x.shape[2]
		# [bs,c,seq_len,n]->[seq_len,n,c]c=2
		x = x.squeeze(0).permute(1,2,0)
		# [seq_len*n,c]
		x = x.reshape(-1,x.shape[-1])
		# [seq_len*n,embedding]
		x = self.fcx(x)
		# [seq_len,n,embedding]
		x = x.reshape(s,n,-1)
		x_gru_h = torch.randn(self.num_layer,n,self.gru_size).cuda()
		_,h = self.grux(x,x_gru_h)
		# [n,embedding]
		x = h.squeeze(0)
		# 重参数化
		e = torch.randn(n,self.latent_size).cuda()
		# 求得先验分布
		p = self.prior(x)
		if(mode=='train'):
			n,s = y.shape[-1],y.shape[2]
			# [bs,c,seq_len,n]->[seq_len,n,c]c=2
			y = y.squeeze(0).permute(1,2,0)
			# [seq_len*n,c]
			y = y.reshape(-1,y.shape[-1])
			# [seq_len*n,embedding]
			y = self.fcy(y)
			# [seq_len,n,embedding]
			y = y.reshape(s,n,-1)
			y_gru_h = torch.randn(self.num_layer,n,self.gru_size).cuda()
			_,y = self.gruy(y,y_gru_h)
			# [n,embedding]
			y = y.squeeze(0)
			# [n,2*embeddin]
			y = torch.cat((x,y),1)
			# 求得后验分布
			q = self.recognition(y)
			z = q[:,0:self.latent_size]+q[:,self.latent_size:]*e
		else:
			z = p[:,0:self.latent_size]+p[:,self.latent_size:]*e
		# [n,gru_size+latent_size]
		x = torch.cat((h.squeeze(0),z),1)
		# 求Goal [n,2]
		g = self.goal(x)
		
		forward_gru_h = self.fc2(x)
		f = self.fcf(x) # 前项输入
		backward_gru_h = self.fc4(x)
		b = self.fc5(g) # 后项输入
		# 计算前向
		forward_output = []
		for i in range(self.pre_len):
			forward_gru_h = self.forward_gru(f,forward_gru_h)
			forward_output.append(forward_gru_h)
			f = self.fc3(forward_gru_h)
			
		# 计算后向
		backward_output = []
		for i in range(self.pre_len-1,-1,-1):
			backward_gru_h = self.backward_gru(b,forward_gru_h)
			temp = torch.cat((forward_output[i],backward_gru_h),1)
			output = self.fc6(temp)
			backward_output.append(output)
			b = self.fc5(output)
			
		if(mode=='train'):
			return p,q,g,torch.stack(backward_output,0).unsqueeze(0)
		else:
			return torch.stack(backward_output,0).unsqueeze(0)

"""
x = torch.randn(1,2,8,24)
y = torch.randn(1,2,12,24)
prior = BiTrap(embedding_size=64,input_size=2,output_size=2,gru_size=256,latent_size=32,gru_input_size=64)
p,q,g,b = prior(x,'train',y)
print(p.shape)
print(q.shape)
print(g.shape)
print(b.shape)

b = prior(x,'test')
print(b.shape)
"""
