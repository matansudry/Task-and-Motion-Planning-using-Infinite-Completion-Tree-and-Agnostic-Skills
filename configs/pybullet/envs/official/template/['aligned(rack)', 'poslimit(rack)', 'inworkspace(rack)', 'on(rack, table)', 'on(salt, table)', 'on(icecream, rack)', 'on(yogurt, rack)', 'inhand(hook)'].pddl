(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
		yogurt - box
		icecream - box
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on salt table)
		(on icecream rack)
		(on yogurt rack)
		(inhand hook)
	)
	(:goal (and))
)
