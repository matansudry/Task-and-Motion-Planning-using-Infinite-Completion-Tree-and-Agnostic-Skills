(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		milk - box
		yogurt - box
		icecream - box
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on icecream rack)
		(on milk table)
		(on yogurt rack)
		(inhand salt)
	)
	(:goal (and))
)
