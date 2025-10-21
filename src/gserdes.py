import io

class Deserializer:
	def __init__(self, stream):
		self.stream = stream
		self.lookahead = self.stream.read(1)

	def peek(self):
		return self.lookahead
	
	def next(self):
		self.lookahead = self.stream.read(1)
		return self.lookahead
	
	def string(self):
		s = io.BytesIO()
		c = self.peek()
		while c != b'"':
			s.write(c)
			c = self.next()
		self.next()
		return s.getvalue()
	
	def white(self):
		c = self.peek()
		while c in b' \t\n':
			c = self.next()
	
	def seqgen(self):
		c = self.peek()
		while c != b')':
			yield self.sexp()
			self.white()
			c = self.peek()
		self.next()
		return
	
	def seq(self):
		return list(self.seqgen())
	
	def sexp(self):
		self.white()
		c = self.peek()
		if c == b'"':
			self.next()
			return self.string()
		elif c == b'(':
			self.next()
			return self.seq()

def serialize(s, stream):
	if isinstance(s, bytes):
		stream.write(b'"')
		stream.write(s)
		stream.write(b'"')
		return
	
	stream.write(b"\n(")
	if len(s) == 0:
		stream.write(b")")
		return 
	s = iter(s)
	serialize(next(s), stream)
	for x in s:
		stream.write(b" ")
		serialize(x, stream)
	stream.write(b")")
