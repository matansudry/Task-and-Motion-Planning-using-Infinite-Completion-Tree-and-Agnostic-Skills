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
		(on hook rack)
		(on milk rack)
		(on yogurt rack)
		(inhand salt)
	)
	(:goal (and))
)
