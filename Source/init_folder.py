import os

def main():
	os.system('rm ./out/coco30bit/encode/*')
	os.system('rm ./result/checkpoints/* -rf')
	os.system('rm output*')
        os.system('mkdir ./out/coco30bit')
        os.system('mkdir ./out/coco30bit/encode')
	
if __name__=='__main__':
	main()
