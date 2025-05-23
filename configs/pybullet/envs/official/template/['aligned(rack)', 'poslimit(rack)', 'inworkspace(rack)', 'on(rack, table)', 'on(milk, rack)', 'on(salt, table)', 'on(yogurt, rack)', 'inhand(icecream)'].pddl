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
		(on milk rack)
		(on salt table)
		(on yogurt rack)
		(inhand icecream)
	)
	(:goal (and))
)
