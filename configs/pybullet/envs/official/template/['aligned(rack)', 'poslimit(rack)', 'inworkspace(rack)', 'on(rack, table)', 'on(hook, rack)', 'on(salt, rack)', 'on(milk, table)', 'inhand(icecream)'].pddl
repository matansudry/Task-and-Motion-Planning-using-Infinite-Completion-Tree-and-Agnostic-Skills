(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
		milk - box
		icecream - box
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on hook rack)
		(on salt rack)
		(on milk table)
		(inhand icecream)
	)
	(:goal (and))
)
