(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
		milk - box
		yogurt - box
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on salt table)
		(on milk table)
		(on yogurt table)
		(inhand hook)
	)
	(:goal (and))
)
