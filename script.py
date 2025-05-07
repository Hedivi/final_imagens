from pathlib import Path



caminho = Path(".")
diretorios_recursivos = [d for d in caminho.rglob("*") if d.is_dir()]

with open('info.csv', 'w') as doc:

	doc.write("path;patient;image_name;roi;x;y;percent;label\n")

	for d in diretorios_recursivos:
		try:
			arquivos = [f for f in d.iterdir() if f.is_file()]
			arqs = [ str(arq).split('/')[1] for arq in arquivos]
	
			for a in arquivos:
				arq = str(a).split('/')
				patient = arq[0]
				arq = arq[1].split('_')
			
				image = arq[0]
				roi = arq[1]
				coordinate = arq[2].split('-')
				x = coordinate[0]
				y = coordinate[1]
				percent = arq[3]
				label = arq[4].split('.')[0]
			
				data  = str(a) + ';'
				data += str(patient) + ';' 
				data += str(image) + ';' 
				data += str(roi) + ';'
				data += str(x) + ';'
				data += str(y) + ';'
				data += str(percent) + ';'
				data += str(label) + '\n'
		
				doc.write(data)
		except:
			print(a)
